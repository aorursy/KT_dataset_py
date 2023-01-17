import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

from matplotlib import rcParams

import seaborn as sns



from catboost import Pool, CatBoostRegressor, cv

from sklearn.model_selection import KFold

from sklearn.metrics import mean_absolute_error as MAE

from sklearn.metrics import mean_squared_error, r2_score
# User functions

DATA_PATH = '../input'

AUX_PATH = '../input/easy-potential-energy-prediction'





def csv_path(dataset="train", data_path=DATA_PATH):

    """

    Get csv data path

    """

    return "{}/{}.csv".format(data_path, dataset)





def read_file(data_file, data_path=DATA_PATH):

    """

    Read csv files from data path

    """

    assert "csv" in data_file

    file_ext = data_file.split(".")[1]

    dataset = data_file.split(".")[0]

    index_col = None

    index_type = ['train', 'test']

    if dataset in index_type:

        index_col = 'id'

    data_path = csv_path(dataset, data_path=data_path)

    return pd.read_csv(data_path, index_col=index_col)





# Atom properties

ATOM_ELECTRONEG = {

    "H": 2.2,

    "C": 2.55,

    "N": 3.04,

    "O": 3.44,

    "F": 3.98

}



ATOM_NUMBER = {

    "H": 1,

    "C": 6,

    "N": 7,

    "O": 8,

    "F": 9

}



# 1 atomic mass unit (amu) corresponds to 1.660539040 × 10−24 gram

ATOM_MASS = {

    "H": 1.00784,

    "C": 12.0107,

    "N": 14.0067,

    "O": 15.9990,

    "F": 18.9984

}



# Without fudge factor

ATOM_RADIUS = {

    'H': 0.38,

    'C': 0.77,

    'N': 0.75,

    'O': 0.73,

    'F': 0.71

}





def d_0(struc_df):

    """

    Atom distance to origin

    struc_df: structure data frame

    """

    return np.sqrt((struc_df.x ** 2) +

                   (struc_df.y ** 2) +

                   (struc_df.z ** 2))





def d_max(struc_df):

    """

    Atom max coord to origin

    df: structure data frame

    """

    return np.max([struc_df.x, struc_df.y, struc_df.z], axis=0)





def d_min(struc_df):

    """

    Atom min coord to origin

    df: structure data frame

    """

    return np.min([struc_df.x, struc_df.y, struc_df.z], axis=0)





def cos_x(struc_df):

    "Atom cos axis x"

    return struc_df.x / d_0(struc_df)





def cos_y(struc_df):

    "Atom cos axis y"

    return struc_df.y / d_0(struc_df)





def cos_z(struc_df):

    "Atom cos axis z"

    return struc_df.z / d_0(struc_df)





def structures_add_features(structures):

    """

    Add new features to "structures.csv"

    """

    atoms_values = structures['atom'].values

    structures["atom_mass"] = [ATOM_MASS[x] for x in atoms_values]

    structures["atom_number"] = [ATOM_NUMBER[x] for x in atoms_values]

    structures["atom_electroneg"] = [ATOM_ELECTRONEG[x] for x in atoms_values]

    structures["atom_radius"] = [ATOM_RADIUS[x] for x in atoms_values]

    structures["atom_d0"] = d_0(structures)

    structures["atom_dmax"] = d_max(structures)

    structures["atom_dmin"] = d_min(structures)

    structures["cos_x"] = cos_x(structures)

    structures["cos_y"] = cos_y(structures)

    structures["cos_z"] = cos_z(structures)

    return structures





def map_atom_info(data_df, structures_df):

    """

    Merge atoms info in structures

    """

    index_name = data_df.index.name

    data_df.reset_index(inplace=True)

    for atom_idx in range(2):

        former_cols = list(data_df)

        data_df = pd.merge(data_df,

                           structures_df,

                           how='left',

                           left_on=['molecule_name', f'atom_index_{atom_idx}'],

                           right_on=['molecule_name', 'atom_index'],

                           left_index=False,

                           right_index=False)

        data_df = data_df.drop('atom_index', axis=1)

        actual_cols = list(data_df)

        new_cols = [x + "_{}".format(atom_idx) for x in actual_cols if x not in former_cols]

        new_cols = former_cols + new_cols

        data_df.columns = new_cols

    data_df.set_index(index_name, inplace=True)

    return data_df





def dist_xyz_2(data_df):

    """

    Coupling Euclidean distance

    """

    return ((data_df.x_1 - data_df.x_0) ** 2 +

            (data_df.y_1 - data_df.y_0) ** 2 +

            (data_df.z_1 - data_df.z_0) ** 2)





def dist_xyz(data_df):

    """

    Coupling Euclidean distance

    """

    return np.sqrt(dist_xyz_2(data_df))





def dist_x(data_df):

    """

    Coupling X axis distance

    """

    return np.sqrt((data_df.x_1 - data_df.x_0) ** 2)





def dist_y(data_df):

    """

    Coupling Y axis distance

    """

    return np.sqrt((data_df.y_1 - data_df.y_0) ** 2)





def dist_z(data_df):

    """

    Coupling Z axis distance

    """

    return np.sqrt((data_df.z_1 - data_df.z_0) ** 2)





def data_add_features(data_df, structures_df):

    """

    Add new features to train or test data frames

    """

    data_df = map_atom_info(data_df, structures_df)

    # Add distance features

    data_df["dist_xyz_2"] = dist_xyz_2(data_df)

    data_df["dist_xyz"] = dist_xyz(data_df)

    data_df["dist_x"] = dist_x(data_df)

    data_df["dist_y"] = dist_y(data_df)

    data_df["dist_z"] = dist_z(data_df)

    # atom_0 is always H

    data_df.drop("atom_0", axis=1, inplace=True)

    return data_df
# kFold Validation Parameters

RANDOM_STATE = 123

N_SPLITS = 3

SHUFFLE = True

VERBOSE = False



# Script parameters

FREE_MEMORY = True

SAVE_SUBMISSION = True

MODEL_NAME = "CB_MULLIKEN_001"

LINE = 40 * "-"



# Ploting Parameters

FIGSIZE = (10, 6)

sns.set()



# Data files

INPUT_FILE_A = "mulliken_charges.csv"

INPUT_FILE_B = "structures.csv"

OUTPUT_FILE = "mulliken_charges_upd.csv"
mulliken_charges = read_file(INPUT_FILE_A)

structures = read_file(INPUT_FILE_B)
mulliken_charges.head()
structures = structures_add_features(structures)

structures.head()
# Basic features

id_feature = 'molecule_name'

target_feature = 'mulliken_charge'
mulliken_train = mulliken_charges.merge(structures)

# Check merged data frame

assert mulliken_train.shape[0] == mulliken_charges.shape[0]

assert mulliken_train.notnull().values.any()
mulliken_test = mulliken_charges.merge(structures,

                                       on=[id_feature, 'atom_index'],

                                       how='right',

                                       indicator=False)



mulliken_test = mulliken_test[mulliken_test[target_feature].isnull()]

mulliken_test = mulliken_test.drop(target_feature, axis=1)

# Check merged data frame

structures_shape = structures.shape

assert mulliken_train.shape[0] + mulliken_test.shape[0] == structures_shape[0]

assert mulliken_test.notnull().values.any()
mulliken_train.tail()
mulliken_test.head()
selected_features = list(mulliken_test)

selected_features.remove(id_feature)

categorical_features = ['atom']

print("Selected Features: \t{}".format(selected_features))

print("Target Feature: \t{}".format(target_feature))

print("Id Feature: \t\t{}".format(id_feature))

print("Categorical Features: \t{}".format(categorical_features))
if FREE_MEMORY:

    del structures, mulliken_charges
X = mulliken_train[selected_features]

y = mulliken_train[target_feature]
kfold = KFold(n_splits=N_SPLITS,

              random_state=RANDOM_STATE,

              shuffle=SHUFFLE)
params = {

    "model_name": MODEL_NAME,

    "iterations": 100,

    "depth": 16,

    "learning_rate": 0.80,

    "reg_lambda": 3.0,

    "loss_function": "RMSE",

    "verbose": VERBOSE,

    "random_seed": RANDOM_STATE,

    "task_type": "GPU"

}



cat_reg = CatBoostRegressor(iterations=params["iterations"],

                            depth=params["depth"],

                            learning_rate=params["learning_rate"],

                            loss_function=params["loss_function"],

                            verbose=params["verbose"],

                            use_best_model=True,

                            reg_lambda=params["reg_lambda"],

                            random_seed=params["random_seed"],

                            task_type=params["task_type"],

                            name=params["model_name"])
%%time

fold = 0

r2_scores = []

mse_scores = []

mae_scores = []





for in_index, oof_index in kfold.split(X, y):

    fold += 1

    print(LINE)

    print("- Training Fold: ({}/{})".format(fold, N_SPLITS))

    X_in, X_oof = X.loc[in_index], X.loc[oof_index]

    y_in, y_oof = y.loc[in_index], y.loc[oof_index]

    # Train and evaluation data pools

    train_pool = Pool(data=X_in,

                      label=y_in,

                      cat_features=categorical_features)

    eval_pool = Pool(data=X_oof,

                     label=y_oof,

                     cat_features=categorical_features)

    # Fit Regressor

    hist = cat_reg.fit(train_pool, eval_set=eval_pool)

    y_pred = cat_reg.predict(X_oof)

    # Metrics

    r2 = r2_score(y_oof, y_pred)

    r2_scores.append(r2)

    mse_score = mean_squared_error(y_oof, y_pred)

    mse_scores.append(mse_score)

    mae_score = MAE(y_oof, y_pred)

    mae_scores.append(mae_score)

    print(f'\t R2:  {r2:.4f}')

    print(f'\t MSE: {mse_score:.4f}')

    print(f'\t MAE: {mse_score:.4f}')

## k-Fold metrics

print('kFold Validation Results:')

print(' * Average Variance Score (R2): \t{:.4f}'.format(np.mean(r2_scores)))

print(' * Average Mean squared error (MSE): \t{:.4f}'.format(np.mean(mse_score)))

print(' * Average Mean absolure error (MAE): \t{:.4f}'.format(np.mean(mae_score)))
# Increase the number of params that matplotlib can handle

rcParams['agg.path.chunksize'] = 10000

# Perfect prediction line

perfect_pred = np.arange(-0.8, 0.9, 0.1)

plt.figure(figsize=FIGSIZE)

plt.plot(perfect_pred , perfect_pred , c="k")

plt.scatter(y_oof, y_pred, s=0.2, alpha=0.1)

plt.title("Fold {} Mulliken Charges Prediction".format(fold))

plt.xlabel("Validation")

plt.ylabel("Predicted")

plt.show()
importances_df = pd.DataFrame({"features":list(X),

                               "importances":cat_reg.feature_importances_})

importances_df = importances_df.sort_values('importances', ascending=True)
importances_df.plot.barh(x='features',

                         y='importances',

                         figsize=FIGSIZE,

                         legend=False,

                         width=0.9)

plt.title("Catboost Regression Feature Importances")

plt.ylabel("")

plt.show()
train_pool = Pool(data=X,

                  label=y,

                  cat_features=categorical_features)



full_cat_reg = CatBoostRegressor(iterations=params["iterations"],

                                 depth=params["depth"],

                                 learning_rate=params["learning_rate"],

                                 loss_function=params["loss_function"],

                                 verbose=params["verbose"],

                                 use_best_model=False,

                                 reg_lambda=params["reg_lambda"],

                                 random_seed=params["random_seed"],

                                 task_type=params["task_type"],

                                 name=params["model_name"])

# Fit Regressor

hist = full_cat_reg.fit(train_pool)
y_test = full_cat_reg.predict(mulliken_test[selected_features])

mulliken_test[target_feature] = y_test
mulliken_train[target_feature].plot.kde(figsize=FIGSIZE, legend=True, label="train")

mulliken_test[target_feature].plot.kde(figsize=FIGSIZE, legend=True, label="test")

plt.title("Train and test Mulliken Charges Density")

plt.ylabel("")

plt.show()
output_features = [id_feature, "atom_index", target_feature]



mulliken_charges_upd = pd.concat([

    mulliken_train[output_features],

    mulliken_test[output_features]],

    ignore_index=True)

    

if FREE_MEMORY:

    del mulliken_train, mulliken_test
assert mulliken_charges_upd.shape[0] == structures_shape[0]

assert mulliken_charges_upd.notnull().values.any()

print(f"Saving file {OUTPUT_FILE}...")

mulliken_charges_upd.to_csv(OUTPUT_FILE, index=False)