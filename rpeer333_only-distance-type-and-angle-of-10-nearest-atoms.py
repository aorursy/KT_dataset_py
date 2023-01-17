%%capture



import os

from os.path import join

import sys

from pprint import pprint



import numpy as np

from numpy.linalg import norm

import pandas as pd

import matplotlib.pyplot as plt



from scipy.spatial import distance_matrix



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error



from lightgbm import LGBMRegressor



import warnings

warnings.filterwarnings('ignore')



DATA_DIR = '../input/champs-scalar-coupling'

ATOMIC_NUMBERS = {

    'H': 1,

    'C': 6,

    'N': 7,

    'O': 8,

    'F': 9

}
structures_dtypes = {

    'molecule_name': 'category',

    'atom_index': 'int8',

    'atom': 'category',

    'x': 'float32',

    'y': 'float32',

    'z': 'float32'

}

structures_df = pd.read_csv(join(DATA_DIR, 'structures.csv'), dtype=structures_dtypes)



structures_df['molecule_index'] = structures_df.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')

structures_df['atom'] = structures_df['atom'].replace(ATOMIC_NUMBERS).astype('int8')

structures_df.index = structures_df.molecule_index

structures_df = structures_df[['atom_index', 'atom', 'x', 'y', 'z']]



structures_df[['x', 'y', 'z']] = structures_df[['x', 'y', 'z']] / 10  # puts all distances approx. in range [0, 1]

print(structures_df.shape)

display(structures_df.head())
def load_j_coupling_csv(file_path: str, train=True, verbose=False):

    train_dtypes = {

        'molecule_name': 'category',

        'atom_index_0': 'int8',

        'atom_index_1': 'int8',

        'type': 'category',

        'scalar_coupling_constant': 'float32'

    }

    df = pd.read_csv(file_path, dtype=train_dtypes)

    df['molecule_index'] = df.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')

    

    if train:

        cols = ['id', 'molecule_index', 'atom_index_0', 'atom_index_1', 'type', 'scalar_coupling_constant']

    else: 

        cols = ['id', 'molecule_index', 'atom_index_0', 'atom_index_1', 'type']

    df = df[cols]



    if verbose:

        print(df.shape)

        display(df.head())

        

    return df

    

train_df = load_j_coupling_csv(join(DATA_DIR, 'train.csv'), verbose=True)



mol2distance_matrix = structures_df.groupby('molecule_index').apply(

    lambda df: distance_matrix(df[['x','y', 'z']].values, df[['x','y', 'z']].values))
def get_center(p0: np.array, p1: np.array) -> np.array:

    return p0 + (p1 - p0)/2





def get_vector(p0: np.array, p1: np.array) -> np.array:

    return p1 - p0





def calc_cosines(coords: np.array, a0: int, a1: int) -> np.array:

    """

    @coordinates:

    atom    x     y     z

    1       1     2     3

    2       4     5     6

    ...

    """

    atom_axes = coords - get_center(coords[a0, :], coords[a1, :])

    main_axis = get_vector(coords[a0, :], coords[a1, :])

    

    dot_products = np.dot(atom_axes, main_axis)

    atom_axes_norms = np.apply_along_axis(norm, 1, atom_axes)

    main_axis_norm = norm(main_axis)

    

    return dot_products / (atom_axes_norms * main_axis_norm)





def get_knn_coordinates(j_coupling: pd.Series,

                        structures=structures_df,

                        mol2dist=mol2distance_matrix,

                        k=10) -> np.array:

    

    a_0, a_1 = j_coupling.atom_index_0, j_coupling.atom_index_1

    mol_df   = structures.loc[j_coupling.molecule_index]

    

    coordinates = mol_df[['x','y', 'z']].values

    a0_coords   = coordinates[a_0, :]

    a1_coords   = coordinates[a_1, :]

    center      = get_center(a0_coords, a1_coords)

    

    cosines = calc_cosines(coordinates, a_0, a_1)

    

    center_distances = distance_matrix(center.reshape(1, 3), coordinates).ravel()

    knn = np.argsort(center_distances)[:(k + 2)]  # atom-indices of KNN-atoms

    

    knn = np.array([x for x in knn if x not in (a_0, a_1)])

    

    distances = center_distances[knn]

    cosines   = cosines[knn]

    types = mol_df.iloc[knn].atom

    

    distances = np.pad(distances, (0, k - len(distances)), 'constant')

    cosines   = np.pad(cosines,   (0, k - len(cosines)),   'constant')

    types     = np.pad(types,     (0, k - len(types)),     'constant')

    

    d_a0_a1 = norm(a1_coords - a0_coords)



    return np.concatenate([[d_a0_a1], distances, cosines, types])





# this may take a while...

id2features = {row.id : get_knn_coordinates(row) for _, row in train_df.iterrows()}
def make_data(df: pd.DataFrame, id2features: dict, random_state=128, split=True):

    tmp_df = df.copy()

    tmp_df['features'] = tmp_df.id.map(id2features)

    

    X = np.stack(tmp_df.features)

    y = tmp_df.scalar_coupling_constant.values



    if split:

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)

        return (X_train, y_train), (X_val, y_val)

    else:

        return X, y



    

# hyper-parameters like in this kernel:

# https://www.kaggle.com/criskiev/distance-is-all-you-need-lb-1-481

LGB_PARAMS = {

        'objective': 'regression',

        'metric': 'mae',

        'verbosity': -1,

        'boosting_type': 'gbdt',

        'learning_rate': 0.2,

        'num_leaves': 128,

        'min_child_samples': 79,

        'max_depth': 9,

        'subsample_freq': 1,

        'subsample': 0.9,

        'bagging_seed': 11,

        'reg_alpha': 0.1,

        'reg_lambda': 0.3,

        'colsample_bytree': 1.0

        }



    

def train_model(train, validation):

    

    X_train, y_train = train

    X_val,   y_val   = validation



    model = LGBMRegressor(**LGB_PARAMS, n_estimators=1500, n_jobs = -1)  # 6000 estimators would be better but take much longer

    model.fit(X_train, y_train,

            eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='mae',

            verbose=100, early_stopping_rounds=200)



    y_pred    = model.predict(X_val)

    score     = np.log(mean_absolute_error(y_val, y_pred))

    residuals = y_val - y_pred

    

    print(f'competition-metric score: {score}')

    return model, score, residuals





def plot_residuals(residuals: np.array):

    plt.hist(residuals, bins=50)

    plt.title('residual distribution')

    plt.show();
# test pipeline with smallest type:



sub_train_df = train_df.query('type == "1JHN"')



(X_train, y_train), (X_val, y_val) = make_data(sub_train_df, id2features)

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)



model, score, residuals = train_model((X_train, y_train), (X_val, y_val))

plot_residuals(residuals)
def cross_validate(df, id2features):

    type2model = {}

    scores = {}



    for type_, type_df in df.groupby('type'):

        print(f'\n\n### {type_}')

        train, validation = make_data(type_df, id2features)

        model, score, residuals = train_model(train, validation)

        type2model[type_] = model

        scores[type_] = score

        plot_residuals(residuals)



    assert len(scores) == len(type2model) == 8

    return  scores





scores = cross_validate(train_df, id2features)
print(f'competition-metric: {np.mean(list(scores.values())):.2f}')

print('scores per type:')

pprint(scores, width=1)
def make_test_data(df: pd.DataFrame, id2features: dict, random_state=128):

    tmp_df = df.copy()

    tmp_df['features'] = tmp_df.id.map(id2features)

    X = np.stack(tmp_df.features)

    return X





test_df = load_j_coupling_csv(join(DATA_DIR, 'test.csv'), train=False, verbose=True)

id2features_test = {row.id : get_knn_coordinates(row) for _, row in test_df.iterrows()}

prediction_df = pd.DataFrame()





for type_ in sorted(train_df.type.unique()):

    print(f'\n### {type_}')

    

    train_type_df = train_df.query('type == @type_')

    X_train, y_train = make_data(train_type_df, id2features, split=False)

    model = LGBMRegressor(**LGB_PARAMS, n_estimators=2000, n_jobs = -1)  # more estimators for test-set

    model.fit(X_train, y_train, eval_metric='mae')

    

    test_type_df = test_df.query('type == @type_')

    X_test = make_test_data(test_type_df, id2features_test)

    y_hat  = model.predict(X_test)

    

    type_pred_df  = pd.DataFrame({'id': test_type_df.id, 'scalar_coupling_constant': y_hat})

    prediction_df = pd.concat([prediction_df, type_pred_df], ignore_index=True)
prediction_df = prediction_df.sort_values('id')

prediction_df.to_csv('submission.csv', index=False)
assert len(prediction_df) == len(test_df)

print(prediction_df.shape)

display(prediction_df.head())