import os, pydicom, random, math

import pandas as pd

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures

from sklearn.cluster import KMeans

from sklearn.linear_model import LinearRegression

from scipy.optimize import minimize



DATA_PATH = "../input/osic-pulmonary-fibrosis-progression/"

N_FOLDS = 5                                   # Number of component models/data splits

MIN_TEST_WEEK = -12

MAX_TEST_WEEK = 133

TARGET_VAR = 'targetFVC'

MAX_SMOOTHING_RANGE = 5                       # Range for observation smoothing



# Image processing

MAX_PLANES, MAX_ROW, MAX_COL = 10, 100, 100   # Maximum dimensions on Z-, X- and Y- axes

INVERSE_WEIGHT_FCT = lambda y: y**2           # Inverse distance penalty function for weighting

N_NEIGHBOURS = 8                              # Number of planes to consider for synthetic planes

PIXEL_VALUE_RANGE = 32747 + 15000



# Model and training hyperparameters

no_img_params = {'shapes': [300,80]}                               # Tabular data model

w_img_params = {'shapes': [700,400,50], 'kernels': 10}             # Tabular+imaging data model

fit_params = {'sample_size': 5000, 'epochs': 10, 'batch_size': 10} # Training



# Confidence model

Z_MOD = 1.1     # Arbitrary modification to credibility coefficient
def combine_duplicates(df, FUN=np.mean):

    

    df = df.assign(patientWeeks = df.Patient + "__" + df.Weeks.astype(str))

    table = df.patientWeeks.value_counts()

    duplicates = table.loc[table > 1].index

    subset = df.loc[df.patientWeeks.isin(duplicates)]

    avgFVC = subset.groupby(['Patient', 'Weeks']).FVC.agg(FUN)

    avgPct = subset.groupby(['Patient', 'Weeks']).Percent.agg(FUN)    

    subset = subset.drop_duplicates(subset=['Patient', 'Weeks']

                                   ).drop(labels=['FVC', 'Percent'], axis=1)

    subset = subset.join(avgFVC, on=['Patient', 'Weeks']

                        ).join(avgPct, on=['Patient', 'Weeks'])

    df = pd.concat([df[~df.patientWeeks.isin(subset.patientWeeks)],

                    subset]).sort_values(by=['Patient', 'Weeks'])

    return df.drop(labels=['patientWeeks'], axis=1)
def weekly_relative_observations(data, delta):

    """

    Outputs a matrix (NumPy Masked Array) of observed relativities.

    """

    limit = delta * 2 + 1

    positions = np.arange(-delta, delta + 1) # Relative weeks of interest

    patients = data.Patient.unique()

    obs_matrix = np.ma.masked_array([], mask=[])   # Observations matrix

    for patient in patients:

        subset = data.loc[data.Patient == patient]

        weeks = np.array(subset.Weeks)

        for i in range(len(weeks)):        

            baseline = weeks[i]

            rel_weeks = weeks - baseline

            obs_exists = np.in1d(positions, rel_weeks) # Reverse mask

            raw_obs_vec = positions.copy() * 0

            relevant_weeks = weeks[np.in1d(rel_weeks, positions)]

            if len(relevant_weeks) < 2:

                continue

            raw_obs_vec[obs_exists] = subset.loc[subset.Weeks.isin(relevant_weeks)].FVC

            obs_vec = np.ma.masked_array([raw_obs_vec], mask=[~obs_exists])

            if len(obs_matrix) == 0:

                obs_matrix = obs_vec

            else:

                obs_matrix = np.ma.concatenate([obs_matrix, obs_vec], axis=0)  

    return obs_matrix





def neg_log_likelihood_approx(mu, x, cov_matrix):

    """

    Objective function to minimize, to smooth observations probabilistically.

    """

    sq_mahalanobis = abs((x - mu).T.dot(np.linalg.inv(cov_matrix)).dot(x - mu))

    return np.log(abs(np.linalg.det(cov_matrix))) + sq_mahalanobis





def probabilistic_smoothing(subset, obs_matrix, maxdelta=5):

    """

    This function outputs a relativity vector, applicable to FVC and Percent 

    observations.

    """      

    weeks = np.array(subset.Weeks)

    FVC = np.array(subset.FVC)

    target_weeks = np.arange(-maxdelta, maxdelta + 1)

    mu = (obs_matrix / obs_matrix[:, maxdelta][:,np.newaxis]).mean(0)

    n, m = len(weeks), max(weeks) - min(weeks) + 1

    relativity_mat = np.zeros((n, m))     

    for i in range(n):

        week = weeks[i]

        rel_weeks = weeks - week

        relevant_obs = np.in1d(rel_weeks, target_weeks)

        x = FVC[relevant_obs] / FVC[i]    

        if len(x) <= 1:

            relativity_mat[i, week - min(weeks)] = 1

        else:

            relevant_params = np.in1d(target_weeks, rel_weeks)

            mu_mod = mu[relevant_params]

            obs_matrix_mod = obs_matrix[:, relevant_params]

            cov_mat = np.ma.cov(obs_matrix_mod, rowvar=False).data

            mu_mod = minimize(neg_log_likelihood_approx, mu_mod,

                              args=(x, cov_mat),

                              method='Nelder-Mead')

            relativity_mat[i, weeks[relevant_obs]-min(weeks)] = mu_mod.x

    positive_rel = relativity_mat > 0

    row_weights = positive_rel * positive_rel.sum(1)[:,np.newaxis]

    keep_cols = relativity_mat.sum(0) > 0

    return np.average(relativity_mat[:, keep_cols], axis=0,

                      weights=row_weights[:, keep_cols])





def smooth_data(df, target_vars, maxrange=MAX_SMOOTHING_RANGE):

    """

    Attempts to reduce the noise in each patient's time series data,

    by smoothing variations in periods of frequent measurements. 

    """

    obs_matrix = weekly_relative_observations(df, maxrange)

    for patient in df.Patient.unique():

        target_loc = df.Patient == patient

        rels = probabilistic_smoothing(df.loc[target_loc], obs_matrix, maxrange)

        for target in target_vars:

            df.at[target_loc, target] = df.loc[target_loc, target] / rels

    return df
def interpolate(df):

    """

    Linearly interpolates the values of FVC and Percent between weeks with data

    """

    ids = np.unique(df.Patient)

    df_mod = pd.DataFrame()

    for i in range(len(ids)):

        subset = df.loc[df.Patient == ids[i]]

        df_mod = pd.concat([df_mod, subset])

        age, sex, smSt = subset.iloc[0].loc[['Age', 'Sex', 'SmokingStatus']]

        for t in range(subset.shape[0]-1):

            gap = subset.Weeks.iloc[t+1] - subset.Weeks.iloc[t]

            if gap > 1:

                base_Week, base_FVC, base_Pct = subset.iloc[t].loc[['Weeks', 'FVC', 

                                                                    'Percent']]

                end_FVC, end_Pct = subset.iloc[t+1].loc[['FVC', 'Percent']]

                for j in range(1, gap):

                    new = pd.DataFrame({'Patient': ids[i],

                                        'Weeks': base_Week + j,

                                        'FVC': base_FVC + j / gap * (end_FVC - base_FVC),

                                        'Percent' : base_Pct + j / gap * (end_Pct - base_Pct),

                                        'Age': age, 'Sex': sex, 'SmokingStatus': smSt

                                       }, index=[None])

                    df_mod = pd.concat([df_mod, new], ignore_index=True)

    return df_mod.sort_values(by=['Patient','Weeks']).reset_index(drop=True)
class CSVDataPrep():

    

    def __init__(self, data_path, nfolds):

        data = pd.read_csv(data_path)

        data = combine_duplicates(data)

        self.raw_data = data

        patients = np.array(data.Patient.unique())

        self.plotter(self.raw_data, patients, True)

        smooth = smooth_data(data.sort_values(by='Weeks'), ['FVC', 'Percent'])

        self.plotter(smooth, patients, False)

        self.data = []

        for df in [data, smooth]:

            # Saves unsmoothed and smoothed observations in positions 0 and 1 respectively

            df = interpolate(df)

            self.data.append(df.assign(targetFVC = df.FVC))

        self.split_folds(data, nfolds)

        plt.tight_layout()

        plt.show()

        

        

    def split_folds(self, data, nfolds):

        nfolds = 2 if nfolds < 2 else int(nfolds)

        patients = np.array(data.Patient.unique())

        valid_size = int(np.round(len(patients) / nfolds))

        self.train_sets, self.valid_sets = [], []

        remaining = patients.copy()

        for i in range(nfolds):

            if i < nfolds - 1:

                self.valid_sets.append(random.sample(list(remaining), valid_size))

            else:

                self.valid_sets.append(remaining)

            self.train_sets.append(patients[~np.in1d(patients, self.valid_sets[i])])

            remaining = remaining[~np.in1d(remaining, self.valid_sets[i])]

        

        

    def pull(self, fold, raw=False):

        """

        Pulls smoothed training and unsmoothed validation data for a single fold.

        

        If raw==True, omits interpolated weeks.

        """

        fold = fold % len(self.train_sets)

        data = self.raw_data

        # Use smoothed data for training; unsmoothed data for validation

        if not raw: data = self.data[1]

        train = data.loc[data.Patient.isin(self.train_sets[fold])]

        if not raw: data = self.data[0]

        valid = data.loc[data.Patient.isin(self.valid_sets[fold])]

        return train, valid

    

    

    def plotter(self, dataset, patients, firstrun, per_row=4):

        n = len(patients)

        color = 'tab:cyan' if firstrun else 'tab:orange'

        label = "Raw" if firstrun else "Smooth"

        if firstrun:

            fig, self.plots = plt.subplots(nrows=int(np.ceil(n/per_row)), ncols=per_row, 

                                           figsize=(12, n//per_row * 2.5))

        for i in np.arange(n):

            patient = patients[i]

            subplot = self.plots[i//per_row, i%per_row]

            subset = dataset.loc[dataset.Patient == patient]

            x, y = np.array(subset.Weeks), np.array(subset.FVC)

            subplot.plot(x, y, color=color, label=label)

            subplot.set_xlabel("Weeks")

            subplot.set_ylabel("FVC")

            subplot.legend(loc="upper right")

        

        

        

csv_data = CSVDataPrep(DATA_PATH + "train.csv", N_FOLDS)
class SamplingMatrix():

    """

    Creates a matrix used for downsampling pixel arrays.

    """    

    def __init__(self, source_size, target_size):

        p = source_size / target_size

        matrix = np.zeros((target_size, source_size))

        for j in range(source_size):

            for i in range(target_size):

                if sum(matrix[:, j]) < 1:

                    matrix[i, j] = min(1, p - sum(matrix[i,:]), 1 - sum(matrix[:,j]))

                else:

                    break

        self.matrix = matrix / p



        

        

class SamplingMatrixContainer():

    """

    Contains already-computed sampling matrices and computes a new one when required.

    """    

    def __init__(self):

        self.matrices = {}

        

        

    def get_matrix(self, source_size, target_size):

        if (source_size, target_size) not in self.matrices.keys():

            self.matrices[(source_size, target_size)] = SamplingMatrix(source_size, 

                                                                       target_size)

        return self.matrices[(source_size, target_size)].matrix





    

class Standardized2DScan():

    """

    This class loads and holds a pixel matrix from a given DICOM file. If the

    matrix contains more rows or columns than the smallest image matrix in our

    dataset, the __init__ method automatically shrinks it so that all matrices

    used by our model have the same size.

    """    

    def __init__(self, filepath, max_dim, sample_matrix_container):

        with pydicom.dcmread(filepath) as f:

            self.pixel_matrix = f.pixel_array

            self.rows, self.cols = f.Rows, f.Columns

            self.thickness, self.i = f.SliceThickness, f.InstanceNumber

        del f

        if self.rows > max_dim[0] or self.cols > max_dim[1]:

            self.shrink_image(max_dim, sample_matrix_container)



            

    def shrink_image(self, max_dim, sample_matrix_container):

        """

        Shrinks a pixel matrix to specified proportions.

        """

        for i in range(2):

            dim = self.pixel_matrix.shape[i]

            if dim > max_dim[i]:

                sampling_matrix = sample_matrix_container.get_matrix(dim, max_dim[i])

                if i == 0: 

                    self.pixel_matrix = sampling_matrix.dot(self.pixel_matrix)

                else: 

                    self.pixel_matrix = sampling_matrix.dot(self.pixel_matrix.T).T

    

    def get_vert_attr(self):

        return self.thickness, self.i

    

    

    

class Standardized3DScan():

    """

    This class loads and holds the scan for a given patient, resizes each matrix if

    necessary. It then creates a collection of equidistant synthetic planes along the

    Z-axis by weighing the pixel values of the actual planes closest to each synthetic

    plane, then stores the result in flat numpy array.

    """    

    def __init__(self, patient, sample_matrix_container, trainpath=True,

                 max_dim=(MAX_PLANES, MAX_ROW, MAX_COL), 

                 inv_weight_fct=INVERSE_WEIGHT_FCT, scaling_factor=PIXEL_VALUE_RANGE): 

        filepath = DATA_PATH + ('train' if trainpath else 'test') + '/' + patient + '/'

        if os.path.isdir(filepath) == False:

            self.voxel_vector = None

            return None

        files = np.array(os.listdir(filepath))

        matrices = {}

        vert_order = pd.DataFrame()

        problem_files, self.error = [], False

        for file in files:

            try:

                matrix = Standardized2DScan(filepath + file, max_dim[1:], 

                                            sample_matrix_container)

                vert_order = vert_order.append({'file': file, 'location': matrix.i,

                                                'thickness': matrix.thickness

                                               }, ignore_index=True)

                matrices[file] = matrix

            except:

                problem_files.append(file)

                

        if len(matrices) < max_dim[0]:

            self.voxel_vector = None

            return None

        else:

            # Deriving approximate .SliceLocation attributes

            ordered = vert_order.sort_values(by='location')

            ordered = ordered.assign(z = np.cumsum(ordered.thickness) - 0.5).sort_index()

            z = np.array(ordered.z)

            # Filter out problem files

            files = files[~np.in1d(files, problem_files)]

            ordered_array = np.concatenate([matrices[file].pixel_matrix 

                                            for file in files[z.argsort()]]

                                          ).reshape(len(files), max_dim[1], max_dim[2])

            z_matrix = self.z_weights(max_dim[0], z, ordered_array, inv_weight_fct)

            voxel_array = np.zeros(np.prod(max_dim)).reshape(max_dim)

            for i in range(max_dim[1]):

                for j in range(max_dim[2]):

                    voxel_array[:, i, j] = z_matrix.dot(ordered_array[:, i, j])

            self.voxel_vector = voxel_array.flatten() / scaling_factor

        del matrices

        

        

    def z_weights(self, num_planes, z_vector, ordered_array, inv_weight_fct, 

                  n_closest_neighbours=N_NEIGHBOURS):

        """

        Returns a matrix containing the weights of each actual plane for computing each 

        synthetic plane.

        """

        if len(ordered_array.shape) != 3:

            return False

        z_range = max(z_vector) - min(z_vector)

        n = ordered_array.shape[0]  # Original number of planes

        weights_matrix = np.zeros((num_planes, n))

        z_pos = np.array([(k+1)*(z_range) 

                          for k in range(num_planes)]) / (num_planes + 1) + min(z_vector)

        for k in range(len(z_pos)):

            z = z_pos[k]

            distances = z_vector - z                                    

            if np.any(distances == 0):

                # No need to interpolate: we have an exact match

                weights_matrix[k, np.where(distances == 0)[0][0]] = 1

            else:

                # Indices of closest points in distances vector

                closest = np.absolute(distances).argsort()  

                target_subset = closest[:n_closest_neighbours]

                if np.all(z_vector[target_subset] > z) or np.all(z_vector[target_subset] < z):

                    # We want to make sure we have at least one weighing point opposite 

                    # to the others

                    # Smallest value greater than target z

                    larger_index = min(np.where(np.sort(z_vector) > z)[0])  

                    # Largest value lesser than target z

                    smaller_index = max(np.where(np.sort(z_vector) < z)[0]) 

                    target_subset = np.concatenate([[larger_index], [smaller_index], 

                                                    target_subset[1:(n_closest_neighbours-1)]])

                subset = distances[target_subset]  # Subset of distances of interest                

                gross_weights = 1 / inv_weight_fct(np.absolute(subset))

                weights_matrix[k, target_subset] = gross_weights / sum(gross_weights)

        return weights_matrix



    



class ScanDataContainer():

    """

    Container for all standardized 3D arrays of scans. If a patient's scans have not yet been 

    processed, the get_voxel_vector() method automatically does it before returning them.

    """

    def __init__(self, training=True, max_dim=(MAX_PLANES, MAX_ROW, MAX_COL), 

                 inv_weight_fct=lambda y: y):

        self.filepath = DATA_PATH + ('train' if training else 'test') + '/'

        self.max_dim = max_dim

        self.inv_weight_fct = inv_weight_fct

        self.images = {}

        self.id_validation = {}

        self.sample_matrix_container = SamplingMatrixContainer()

        

    def validate_availability(self, patients, trainpath=True):

        for patient in patients:

            if patient not in self.id_validation.keys():

                validation = self.get_voxel_vector(patient, trainpath)

                self.id_validation[patient] = validation is not None

        return {p: self.id_validation[p] for p in patients}

    

    def get_voxel_vector(self, patient, trainpath=True):

        if patient not in self.images.keys():

            self.images[patient] = Standardized3DScan(patient, self.sample_matrix_container, 

                                                      trainpath=trainpath)

        return self.images[patient].voxel_vector

    

    

    

img_data =  ScanDataContainer()
class CustomClusterer():

    """

    This class considers every training voxel vector instance as a centroid, and

    transforms new image data as vectors of euclidian distances to each centroid.

    """

    def __init__(self, img_data):

        self.clusters = []

        self.img_data = img_data

        

        

    def fit(self, voxel_mat):

        self.clusters = voxel_mat

        

        

    def transform(self, voxel_mat):

        result_matrix = np.array([])

        for cluster in self.clusters:

            distance = lambda vec1, vec2: np.linalg.norm(vec1 - vec2)

            eucl_dist_vec = np.apply_along_axis(distance, 1, voxel_mat, cluster)

            if result_matrix.shape[0] == 0:

                result_matrix = eucl_dist_vec[:,np.newaxis]

            else:

                result_matrix = np.concatenate([result_matrix, 

                                                eucl_dist_vec[:,np.newaxis]

                                               ], axis=1)

        return result_matrix

    

    

    

class ScanDataClusterer():

    """

    This class repackages voxel data using the K-means algorithm OR the 

    CustomClusterer class. Outputs a DataFrame with the euclidian

    distances between observations and each cluster.

    """    

    def __init__(self, img_data, kernels=0):

        self.img_data = img_data

        self.n_kernel = kernels

        if kernels <= 0:

            self.kernelizer = CustomClusterer(img_data)

        else:

            self.kernelizer = KMeans(n_clusters=kernels)

        self.scaler = StandardScaler()

        

    

    def fit(self, patients):

        voxel_mat = np.array([self.img_data.get_voxel_vector(p)

                              for p in np.unique(patients)])

        voxel_mat = self.scaler.fit_transform(voxel_mat)

        self.kernelizer.fit(voxel_mat)

        

        

    def fit_transform(self, data):

        self.fit(np.array(data.Patient))

        return self.transform(data)

        

        

    def transform(self, newdata, trainpath=True):

        patients = np.array(newdata.Patient)

        voxel_mat = np.array([self.img_data.get_voxel_vector(p, trainpath=trainpath)

                              for p in np.unique(patients)])

        if len(voxel_mat.shape) == 1: 

            voxel_mat = voxel_mat.reshape(1, -1)    

        voxel_mat = self.scaler.transform(voxel_mat)

        voxel_df = pd.DataFrame(self.kernelizer.transform(voxel_mat))

        

        voxel_df.columns = [f'dist_k{k}' for k in range(len(voxel_df.columns))]

        voxel_data = pd.DataFrame({'Patient': np.unique(patients)}).join(voxel_df)

        indices = [voxel_data.loc[voxel_data.Patient == p].index[0] 

                   for p in patients]

        return voxel_data.iloc[indices].reset_index(drop=True).drop('Patient', axis=1)
class CustomModel():

    """

    This class defines an underlying neural network for predictions, to be fed

    fully preprocessed data.

    """

    def __init__(self, shapes=[50,10]):

        self.model = tf.keras.Sequential()

        self.shapes = shapes

        

    def fit(self, X_train, y_train, eval_set, batch_size=50, epochs=2, 

            callbacks=[], verbose=0, dropout=0):

        self.model.add(tf.keras.Input(shape=(X_train.shape[1],)))

        for shape in self.shapes:

            self.model.add(tf.keras.layers.Dropout(dropout, input_shape=(shape,)))

            self.model.add(tf.keras.layers.Dense(shape, activation="elu"))

        self.model.add(tf.keras.layers.Dense(1))     

        optimizer = tf.keras.optimizers.Adam()

        self.model.compile(optimizer = optimizer,

                           loss = tf.keras.losses.MeanSquaredLogarithmicError(),

                           metrics = [tf.keras.metrics.MeanSquaredLogarithmicError()],)

        history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,

                                 validation_data=eval_set[0], #callbacks=callbacks, 

                                 verbose=verbose)

        self.best_score = self.model.evaluate(*eval_set[0], batch_size=200)

        

    def predict(self, X):

        return self.model.predict(X)



        

    def summary(self):

        self.model.summary()



        

        

class FibrosisModel():

    """

    This class encapsulates the underlying model as well as all preprocessing objects

    and methods.

    """

    def __init__(self, model_type, img_data=None, kernels=None, 

                 y=TARGET_VAR, **kwargs):

        self.y = y

        self.model = model_type(**kwargs)

        self.img_data = img_data

        if img_data is not None and kernels is not None:

            self.kernelizer = ScanDataClusterer(img_data, kernels=kernels)

        else:

            self.kernelizer = None

        self.cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

        self.scaler = StandardScaler()

        self.firstrun = True

        

        

    def sampler(self, data, sample_size):

        data = data.reset_index(drop=True)

        output = pd.DataFrame()

        big_urn = np.array(data.index)

        for i in range(sample_size):

            choice = random.choice(big_urn)

            draw = data.iloc[choice]

            patient = draw.Patient

            curWeek = draw.Weeks            

            small_urn = np.array(data.loc[(data.Patient == patient) &

                                          (data.Weeks != curWeek)].index)

            choice = random.choice(small_urn)

            target = data.iloc[choice]

            draw.at[self.y] = target.loc[self.y]

            output = output.append({**draw, 'targetWeek':target.Weeks

                                   }, ignore_index=True)

        return output.reset_index(drop=True)

    

    

    def split_from_target(self, data):

        return data.loc[:,~data.columns.isin(['Patient', self.y])], data.loc[:, self.y]

    

    

    def preprocess(self, data, trainpath=True, img_data=None,

                   cat_vars=['Sex','SmokingStatus'], scale_vars=['Percent','Age']):  

        data = data.reset_index(drop=True)

        if self.firstrun:

            scale_cols = pd.DataFrame(self.scaler.fit_transform(data[scale_vars]),

                                      index=data.index)

            cat_cols = pd.DataFrame(self.cat_encoder.fit_transform(data[cat_vars]), 

                                    index=data.index)

            if self.kernelizer is not None: 

                img_cols = self.kernelizer.fit_transform(data)

            self.firstrun = False

        else:

            scale_cols = pd.DataFrame(self.scaler.transform(data[scale_vars]), 

                                      index=data.index)

            cat_cols = pd.DataFrame(self.cat_encoder.transform(data[cat_vars]), 

                                    index=data.index) 

            if self.kernelizer is not None: 

                img_cols = self.kernelizer.transform(data, trainpath=trainpath)

                

        if self.img_data is not None and self.kernelizer is None:

            img_cols = pd.DataFrame([self.img_data.get_voxel_vector(p, trainpath=trainpath)

                                     / PIXEL_VALUE_RANGE

                                     for p in np.array(data.Patient)])

            img_cols.columns = ["v" + str(i) for i in range(img_cols.shape[1])]

                

        scale_cols.columns = scale_vars

        cat_cols.columns = [s.replace(" ","").replace("-","") 

                            for s in np.concatenate(self.cat_encoder.categories_)]

        data = data.drop(np.concatenate([cat_vars, scale_vars]), axis=1)

        data = pd.concat([data, scale_cols, cat_cols], axis=1)

        if self.img_data is not None:

            data = data.join(img_cols)

        return data.loc[:,np.sort(data.columns)]

    

    

    def remove_invalid(self, data, trainpath=True):

        if self.img_data is not None:

            patients = np.array(data.Patient.unique())

            availability = self.img_data.validate_availability(patients, trainpath)

            patients = patients[[availability[p] for p in patients]]

            return data.loc[data.Patient.isin(patients)]

        else:

            return data

        

        

    def fit(self, train, valid, sample_size=1000, plot=False, early_stop=5, 

            verbose=False, callbacks=[], **kwargs):  

        train, valid = self.remove_invalid(train), self.remove_invalid(valid)

        train, valid = self.sampler(train, sample_size), self.sampler(valid, sample_size)

        train, valid = self.preprocess(train), self.preprocess(valid)

        X_train, y_train = self.split_from_target(train)

        X_valid, y_valid = self.split_from_target(valid)

        self.model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],

                       verbose=verbose, #callbacks=callbacks, 

                       **kwargs)

            

    def predict(self, newdata, trainpath=True):

        newdata = self.preprocess(newdata, trainpath)

        newdata = newdata.drop("Patient", axis=1)

        if TARGET_VAR in newdata.columns:

            newdata = newdata.drop(TARGET_VAR, axis=1)

        return self.model.predict(newdata)

    

    

    

class CustomEnsembleModel():

    """

    This class creates an ensemble of models using k-folds validation

    and averaging.

    """    

    def __init__(self, csv_data, no_img_params, w_img_params, fit_params,

                 img_data=img_data, nfolds=N_FOLDS):

        self.csv_data = csv_data

        self.img_data = img_data

        self.nfolds = nfolds

        self.models = {True: [], False: []}

        self.model_params = {True: w_img_params,

                             False: no_img_params}

        self.fit_params = fit_params

        self.score = {}

        

    

    def evaluate(self, valid_set, with_imgs=False):

        model = self.models[with_imgs]

        rmse_vec = []

        for i in range(self.nfolds):

            X = model[i].remove_invalid(valid_set[i])

            labels = np.array(X.loc[:,TARGET_VAR])

            preds = model[i].predict(X)

            if len(labels.shape) > 1:

                labels = np.squeeze(labels)

            if len(preds.shape) > 1:

                preds = np.squeeze(preds)

            rmse_vec.append(np.mean((labels - preds)**2)**0.5)

        self.score[with_imgs] = np.mean(rmse_vec)

        return self.score[with_imgs]

        

        

    def fit(self, with_imgs=False, verbose=0, callbacks=[]):

        model_set = self.models[with_imgs]

        params = self.model_params[with_imgs]

        img_data = self.img_data if with_imgs else None

        

        for i in range(self.nfolds):

            print("\nTraining model", (i+1),"/",self.nfolds,

                  ("with" if with_imgs else "without"),"images")

            train, valid = self.csv_data.pull(i)

            model = FibrosisModel(CustomModel, img_data=img_data, **params)

            model.fit(train, valid, verbose=verbose, #callbacks=callbacks, 

                      **self.fit_params)

            while model.model.best_score[0] > 0.1:

                model = FibrosisModel(CustomModel, img_data=img_data, **params)

                model.fit(train, valid, **self.fit_params)

            model_set.append(model)

            

            

    def predict(self, newdata, with_imgs=False, trainpath=True, folds=None):

        pred_matrix = np.array([])

        model_set = self.models[with_imgs]

        if folds is None:

            folds = range(len(model_set))

        for fold in folds:

            pred_vec = model_set[fold].predict(newdata, trainpath=trainpath)

            if len(pred_vec.shape) == 1:

                pred_vec = pred_vec[:, np.newaxis]

            if len(pred_matrix.shape) == 1:

                pred_matrix = pred_vec

            else:

                pred_matrix = np.concatenate([pred_matrix, pred_vec], axis=1)

        try:

            result = pred_matrix.mean(1)

        except:

            print("CustomEnsembleModel.predict() L171")

            print(pred_matrix.shape)

            result = pred_matrix

        return pred_matrix.mean(1)



    

model = CustomEnsembleModel(csv_data, no_img_params, w_img_params, fit_params)

model.fit(False) # Train the version with only tabular data

model.fit(True)  # Train the version that also uses imaging data
class ConfidenceModel():

    

    def __init__(self, prediction_models, tabular_data, img_data):

        self.prediction_models = prediction_models

        self.tabular_data = tabular_data

        self.img_data = img_data

        self.get_absolute_log_errors()

        self.init_regressors()

        self.compute_experience_params()

        print("Done.")

        plt.show()

    

    

    def get_absolute_log_errors(self, folds=N_FOLDS):

        print("Computing known absolute log-errors...")

        data = self.tabular_data

        # Prepare sets of predictions to make

        pred_sets = []

        whole_set = pd.DataFrame()

        for fold in np.arange(folds):

            _, valid = data.pull(fold, raw=True)

            pred_set = pd.DataFrame()

            for i, row in valid.iterrows():

                subset = valid.loc[valid.Patient == row.Patient]

                subset = subset.assign(targetWeek = subset.Weeks,

                                       targetFVC = subset.FVC)

                subset.Weeks = int(row.Weeks)

                subset.FVC = float(row.FVC)

                subset.Percent = float(row.Percent)

                subset = subset.loc[subset.Weeks!=subset.targetWeek]

                pred_set = pd.concat([pred_set, subset], ignore_index=True)

            pred_sets.append(pred_set)     

            whole_set = pd.concat([whole_set, pred_set], ignore_index=True)

        pred_sets.append(whole_set)

        # Compute absolute log errors

        self.error_sets = {}

        models = self.prediction_models

        for img_context in [False, True]:

            abs_log_error_set = []

            for fold in np.arange(folds + 1):

                pred_set = pred_sets[fold]

                if img_context:

                    with_imgs = self.img_data.validate_availability(np.array(pred_set.Patient))

                    valid = [with_imgs[patient] for patient in pred_set.Patient]

                    pred_set = pred_set.loc[valid]

                preds = models.predict(pred_set, with_imgs=img_context,

                                       folds=([fold] if fold < folds else None))

                abs_log_errors = np.absolute(np.log(preds / pred_set.targetFVC))

                ale_df = pd.DataFrame({'delta': np.array(pred_set.targetWeek - pred_set.Weeks),

                                       'abs_log_error': abs_log_errors})

                abs_log_error_set.append(ale_df)

            self.error_sets[img_context] = abs_log_error_set

            

            

    def init_regressors(self):

        print("Initializing regressors...")

        self.regs = {}

        self.limits = [0,1]

        for img_context in [True, False]:

            regs = []

            for error_set in self.error_sets[img_context]:

                reg = LinearRegression()

                x = np.array(error_set.delta)

                y = np.array(error_set.abs_log_error).reshape(-1,1)

                if min(x) < self.limits[0]: self.limits[0] = min(x)

                if max(x) >= self.limits[1]: self.limits[1] = max(x) + 1

                x = np.stack([x, x**2, x**3], axis=-1)

                reg.fit(x, y)

                regs.append(reg)

            self.regs[img_context] = regs

            

        

    def compute_experience_params(self):

        print("Computing Bulhmann-Straub credibility parameters...")

        limit = -self.limits[0]

        deltas = np.delete(np.arange(*self.limits), limit)

        self.z = {}

        for img_context in [True,False]:

            # Compute weights and average values by component/time-delta

            weights = np.zeros((N_FOLDS, len(deltas)))

            X = np.zeros((N_FOLDS, len(deltas)))

            for fold in np.arange(N_FOLDS):

                for delta in deltas:

                    error_set = self.error_sets[img_context][fold]

                    j = delta + limit + (-1 if delta>0 else 0)

                    weights[fold, j] = sum(error_set.delta == delta)

                    X[fold, j] = error_set[error_set.delta == delta].abs_log_error.mean()

            X[np.isnan(X)] = 0

            # Compute nonparametric Buhlmann-Straub quantities

            Xww = np.average(X, weights=weights)

            Xiw = np.average(X, weights=weights, axis=0)

            Wiw = weights.sum(0)

            s2 = (weights * (X - Xiw)**2).sum() / sum(Wiw-1)

            a = sum(Wiw * (Xiw - Xww)**2) - (X.shape[1]-1) * s2

            a = a * weights.sum() / (weights.sum()**2 - sum(Wiw**2))

            z = Wiw / (Wiw + s2 / a)

            z_df = pd.DataFrame({'delta': deltas, 'z': z})

            self.z[img_context] = z_df

            plt.plot(z_df.delta, z_df.z, color=('magenta' if img_context else 'cyan'),

                     label=('Tabular + imaging data' if img_context else 'Tabular data'))

        plt.xlabel("Weeks delta")

        plt.ylabel("Credibility coefficient")

        plt.legend(loc="center")

        

        

    def predict(self, start_week, img_avail=False, fwd=True, z_mod=Z_MOD):

        regs, z = self.regs[img_avail], self.z[img_avail] # Get the right models     

        # Time deltas of interest

        mod = 1 if fwd else -1

        weeks = np.arange(start_week, (MAX_TEST_WEEK if fwd else MIN_TEST_WEEK), mod) + mod

        deltas = weeks - start_week  

        # Credibility coefficients

        coefs = np.array([(float(z.loc[z.delta==d].z) 

                           if d in np.array(z.delta) else 0)

                          for d in deltas]) * z_mod

        coefs = np.stack([coefs, 1 - coefs], axis=-1)

        # Predictions

        deltas = np.stack([deltas, deltas**2, deltas**3], axis=-1)

        ens_pred = self.regs[img_avail][-1].predict(deltas)

        comp_pred = np.concatenate([self.regs[img_avail][i].predict(deltas)

                                    for i in np.arange(N_FOLDS)], axis=1).mean(1)

        composite = np.concatenate([ens_pred, comp_pred.reshape(-1,1)], axis=1)

        return np.exp((coefs * composite).sum(1)) - 1

        

            

confidence_model = ConfidenceModel(model, csv_data, img_data)
def predict_all(model, data, img_data, confidence, trainpath=False,

                lower=MIN_TEST_WEEK, upper=MAX_TEST_WEEK, default_conf=70):

    

    drop_at_the_end = [c for c in data.columns if c not in ['Weeks', 'Patient']]

    weeks = np.arange(lower, upper+1)

    output = pd.DataFrame()

    data = data.assign(targetWeek = data.Weeks,

                       targetFVC = data.FVC,

                       Confidence = default_conf)

    

    # Visualization of results

    per_row = 3

    n = data.shape[0]

    fig, axes = plt.subplots(nrows=int(np.ceil(n/per_row)), ncols=per_row,

                             figsize=(12, n//per_row * 4))

    

    for idx, row in data.iterrows():

        patient, start_week, start_FVC = row.Patient, row.Weeks, row.FVC

        if len(output) > 0:

            # To avoid any duplicates

            if patient in np.array(output.Patient):

                continue

        

        with_imgs = img_data.validate_availability([patient], trainpath=trainpath)

        with_imgs = with_imgs[patient] # .validate_availability returns a dict

            

        past, future = weeks[weeks < start_week], weeks[weeks > start_week]

        

        pre_output = data.iloc[np.repeat(idx, len(weeks))].reset_index(drop=True)

        pre_output.at[:, 'targetWeek'] = weeks

        

        for timerange in [past, future]:

            if len(timerange) > 0:

                if len(future) > 0:

                    ascending = timerange[0] == future[0]

                else:

                    ascending = False

                    

                target_subset = pre_output.targetWeek.isin(timerange)

                pred_subset = pre_output.loc[target_subset]

                pred_subset = pred_subset.sort_values(by='targetWeek', ascending=ascending)

                pred_subset = pred_subset.loc[:,~pre_output.columns.isin(['targetFVC', 

                                                                          'Confidence'])]

                

                # Take preds and confidence of imageless model in all cases

                preds_no_img = model.predict(pred_subset, with_imgs=False, trainpath=trainpath)

                conf_no_img = confidence.predict(start_week, fwd=ascending)

                conf_no_img = np.maximum(1, np.absolute(conf_no_img * preds_no_img))

                

                # If imaging is available, average predictions from model with and without images,

                # weighted by their respective confidence

                if with_imgs:

                    preds_w_img = model.predict(pred_subset, with_imgs=True, trainpath=trainpath)

                    conf_w_img = confidence.predict(start_week, img_avail=True, fwd=ascending)

                    conf_w_img = np.maximum(1, np.absolute(conf_w_img * preds_w_img))

                    # Weighting

                    weights = np.stack((conf_no_img, conf_w_img), axis=-1) ** -2

                    weights = weights / weights.sum(1, keepdims=True)

                    preds = (weights * np.stack((preds_no_img, preds_w_img), axis=-1)).sum(1)

                    conf = (weights * np.stack((conf_no_img, conf_w_img), axis=-1)).sum(1) # Not quite rigorous but close enough

                else:

                    preds = preds_no_img

                    conf = conf_no_img                    

                        

                if len(preds.shape) > 1:

                    preds = preds[:,0]

                if not ascending:

                    preds = preds[::-1]

                    conf = conf[::-1]

                pre_output.at[target_subset,'targetFVC'] = preds

                mod_conf = np.minimum(1000, conf) * 2 ** 0.5

                pre_output.at[target_subset,'Confidence'] = mod_conf

                

                

        

        output = pd.concat([output, pre_output]).reset_index(drop=True)

        

        # Viz

        if n <= per_row:

            this_plot = axes[idx]

        else:

            this_plot = axes[idx//per_row, idx%per_row]

        this_plot.set_xlim(lower, upper)

        this_plot.set_ylim(0,4000)

        this_plot.set_xlabel("Weeks")

        this_plot.set_ylabel("Predicted FVC")

        this_plot.set_title(patient)

        x, y = pre_output.targetWeek, pre_output.targetFVC

        yl, yu = y - pre_output.Confidence, y + pre_output.Confidence

        this_plot.plot(x, yl, x, yu, color='cyan', linestyle='dashed')

        this_plot.fill_between(x, yl, yu, where=yu>yl, facecolor='cyan', 

                               interpolate=True, alpha=0.5)

        this_plot.plot(x, y, color='blue')

        this_plot.axvline(x=start_week, color='k', linestyle='dashed')

    

    output = output.drop(drop_at_the_end, axis=1)

    

    plt.tight_layout()

    plt.show()

        

    return output





def finalize_format(df):

    df = df.assign(Patient_Week = df.Patient + '_' + df.targetWeek.astype(str))

    df = df.rename(columns={'targetFVC':'FVC'})

    return df.drop(['Patient', 'targetWeek', 'Weeks'], axis=1)





test_data = pd.read_csv(DATA_PATH + "test.csv")

output = predict_all(model, test_data, img_data, confidence_model)

final = finalize_format(output)

final.to_csv("submission.csv", index=False)

final.head()