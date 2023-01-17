%ls ../input
dataset_dir = '/kaggle/input/'

download_dir = './'
is_sample = False # if True, run in test mode

boosting_rounds = 15000 # lightgbm training epochs

boruta_max_iter = 50 # max iteration number for boruta

num_boruta_rows = 5000 # use a small subsample to quickly fit with boruta feature selector
import numpy as np

import pandas as pd

import featuretools as ft

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn import metrics

import gc

import lightgbm

from sklearn.ensemble import RandomForestRegressor

from boruta import BorutaPy

from dask.distributed import LocalCluster



import warnings

warnings.filterwarnings('ignore')
# calculate competition metric

def competition_metric(df, preds, verbose=0):

    # log of mean absolute error, calculated for each scalar coupling type.

    df_copy = df.copy()

    df_copy["prediction"] = preds

    maes = []

    for t in df_copy.type.unique():

        y_true = df_copy[df.type == t].scalar_coupling_constant.values

        y_pred = df_copy[df.type == t].prediction.values

        mae = np.log(metrics.mean_absolute_error(y_true, y_pred))

        if verbose == 1:

            print(f"{t} log(MAE): {mae}")

        maes.append(mae)

    del df_copy

    gc.collect()

    return np.mean(maes)
train = pd.read_csv(f"{dataset_dir}train.csv")

train.head()
test = pd.read_csv(f"{dataset_dir}test.csv")

test.head()
concat = pd.concat([train, test])
structures = pd.read_csv(f"{dataset_dir}structures.csv")

structures.head()
# map structures dataframe into concat

def map_atom_info(df, atom_idx):

    df = pd.merge(df, structures, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    df = df.drop('atom_index', axis=1)

    df = df.rename(columns={'atom': f'atom_{atom_idx}',

                            'x': f'x_{atom_idx}',

                            'y': f'y_{atom_idx}',

                            'z': f'z_{atom_idx}'})

    return df



concat = map_atom_info(concat, 0)

concat = map_atom_info(concat, 1)



concat.head()
# create basic features like distance

def particle_distance(df):

    dist = ( (df["x_1"] - df["x_0"])**2 + (df["y_1"] - df["y_0"])**2 + (df["z_1"] - df["z_0"])**2 )**0.5

    return dist



concat["distance"] = particle_distance(concat)



# create distance values for each axis

def particle_distance_x(df):

    dist = ( (df["x_1"] - df["x_0"])**2 )**0.5

    return dist



def particle_distance_y(df):

    dist = ( (df["y_1"] - df["y_0"])**2 )**0.5

    return dist



def particle_distance_z(df):

    dist = ( (df["z_1"] - df["z_0"])**2 )**0.5

    return dist



concat["distance_x"] = particle_distance_x(concat)

concat["distance_y"] = particle_distance_y(concat)

concat["distance_z"] = particle_distance_z(concat)



concat.head()
if is_sample:

    print("\n!!! WARNING SAMPLE MODE ACTIVE !!!\n")

    concat = concat[:1000]
le = preprocessing.LabelEncoder()

mol_atom_0 = concat.molecule_name.astype(str) + '_' + concat.atom_index_0.astype(str)

concat['molecule_atom_0_id'] = le.fit_transform(mol_atom_0)
le = preprocessing.LabelEncoder()

mol_atom_1 = concat.molecule_name.astype(str) + '_' + concat.atom_index_1.astype(str)

concat['molecule_atom_1_id'] = le.fit_transform(mol_atom_1)
concat.head()
# Create the entity set for featuretools

es = ft.EntitySet(id='concat')
# Add entites to entity set

es = es.entity_from_dataframe(

    entity_id='concat', dataframe=concat.drop(['scalar_coupling_constant'], axis=1), index='id')
es = es.normalize_entity(

    base_entity_id='concat',

    new_entity_id='molecule_atom_0',

    index='molecule_atom_0_id',

    additional_variables=['atom_0', 'x_0', 'y_0', 'z_0'])
es = es.normalize_entity(

    base_entity_id='concat',

    new_entity_id='molecule_atom_1',

    index='molecule_atom_1_id',

    additional_variables=['atom_1', 'x_1', 'y_1', 'z_1'])
es
# It is faster when using n_jobs > 1, however kaggle kernels die if I define multiple jobs, so I comment out those lines below.

#cluster = LocalCluster()
%%time



# Perform an automated Deep Feature Synthesis with a depth of 2

#features0, feature_names0 = ft.dfs(entityset=es, target_entity='molecule_atom_0', max_depth=2, dask_kwargs={'cluster': cluster}, n_jobs=2)

features0, feature_names0 = ft.dfs(entityset=es, target_entity='molecule_atom_0', max_depth=2)

print(features0.shape)
%%time



# Perform an automated Deep Feature Synthesis with a depth of 2

#features1, feature_names1 = ft.dfs(entityset=es, target_entity='molecule_atom_1', max_depth=2, dask_kwargs={'cluster': cluster}, n_jobs=2)

features1, feature_names1 = ft.dfs(entityset=es, target_entity='molecule_atom_1', max_depth=2)

print(features1.shape)
feature_names0
feature_names1
# add column suffixes

def col_suffix_handler(df, suffix):

    col_dict = {col:"{}{}".format(col, suffix) for col in df.columns.values}

    df.rename(columns=col_dict, inplace=True)

    return df



# I will need unqiue feature names after feature selection with boruta

features0 = col_suffix_handler(features0, '__molecule_atom_0')

features1 = col_suffix_handler(features1, '__molecule_atom_1')
# reduce memory

def reduce_memory(df):

    num_converted_cols = 0

    for col in df.columns.values:

        if df[col].dtype == "float64":

            num_converted_cols += 1

            df[col] = df[col].astype("float32")

        elif df[col].dtype == "int64":

            num_converted_cols += 1

            df[col] = df[col].astype("int32")

    print("{} cols converted.".format(num_converted_cols))

    return df



concat = reduce_memory(concat)

features0 = reduce_memory(features0)

features1 = reduce_memory(features1)
# handle NaN values

def nan_handler(df):

    for col in df.columns.values:

        if np.any(df[col].isnull()):

            print(col)

            if df[col].dtype == 'O':

                df[col] = df[col].fillna('NO_VALUE')

            else:

                df[col] = df[col].fillna(-999)

    return df

                

features0 = nan_handler(features0)

features1 = nan_handler(features1)
# handle inf/-inf values

def inf_handler(df):

    for col in df.columns.values:

        if np.any(df[col]==np.inf) or any(df[col]==-np.inf):

            print(col)

            if df[col].dtype == 'O':

                df[df[col]==np.inf] = 'NO_VALUE'

                df[df[col]==-np.inf] = 'NO_VALUE'

            else:

                df[df[col]==np.inf] = 999

                df[df[col]==-np.inf] = 999

    return df

                

features0 = inf_handler(features0)

features1 = inf_handler(features1)
# list unnecessary columns

cols_to_remove = [

    'id',

    'scalar_coupling_constant'

]
%%time



# feature selection using boruta



# merge features with concat df

concat_features_ = concat.iloc[:num_boruta_rows].merge(

    features0, left_on=['molecule_atom_0_id'], right_index=True, how='left')



concat_features_ = concat_features_.iloc[:num_boruta_rows].merge(

    features1, left_on=['molecule_atom_1_id'], right_index=True, how='left')



# label encode object type (categorical) columns

for col in concat_features_.columns.values:

    if concat_features_[col].dtype == 'O':

        le = preprocessing.LabelEncoder()

        concat_features_[col] = le.fit_transform(concat_features_[col])



forest = RandomForestRegressor(n_jobs=-1)



feat_selector = BorutaPy(

    forest, n_estimators='auto', verbose=2, random_state=42, max_iter=boruta_max_iter, perc=90)



X = concat_features_.drop(cols_to_remove, axis=1).iloc[:num_boruta_rows, :].values

y = concat_features_[["scalar_coupling_constant"]].values[:num_boruta_rows, 0]



feat_selector.fit(X, y)



features = concat_features_.drop(cols_to_remove, axis=1).columns.values.tolist()



del X, y, concat_features_

gc.collect()



# list selected boruta features

selected_features = []

indexes = np.where(feat_selector.support_ == True)

for x in np.nditer(indexes):

    selected_features.append(features[x])



print(len(selected_features))

print(selected_features)
# merge features0 and features1 with concat df (using only selected features)



selected_features0_ = list(set(selected_features) - set(concat.columns.values.tolist()))

selected_features0_ = [f for f in selected_features0_ if '__molecule_atom_0' in f]



selected_features1_ = list(set(selected_features) - set(concat.columns.values.tolist()))

selected_features1_ = [f for f in selected_features1_ if '__molecule_atom_1' in f]



concat_features = concat.merge(

    features0[selected_features0_], 

    left_on=['molecule_atom_0_id'], 

    right_index=True, 

    how='left'

)



concat_features = concat_features.merge(

    features1[selected_features1_], 

    left_on=['molecule_atom_1_id'], 

    right_index=True,

    how='left'

)



print(concat_features.shape)

concat_features.head()
concat_features.dtypes.unique()
# label encode object type columns

for col in concat_features.columns.values:

    if concat_features[col].dtype == 'O':

        le = preprocessing.LabelEncoder()

        concat_features[col] = le.fit_transform(concat_features[col])

        

concat_features.head()
len_train = len(train)

del train, test, concat, features0, features1

gc.collect()
train = concat_features[:len_train]

test = concat_features[len_train:]
del concat_features

gc.collect()
%%time

# use selected boruta features and train a lightgbm model



train_index, valid_index = train_test_split(np.arange(len(train)),random_state=42, test_size=0.1)



X_train = train[selected_features].values[train_index]

y_train = train[['scalar_coupling_constant']].values[:, 0][train_index]



valid_df = train.iloc[valid_index]



del train

gc.collect()



X_valid = valid_df[selected_features].values

y_valid = valid_df[['scalar_coupling_constant']].values[:, 0]



params = {'boosting': 'gbdt', 'colsample_bytree': 1, 

              'learning_rate': 0.1, 'max_depth': 40, 'metric': 'mae',

              'min_child_samples': 50, 'num_leaves': 500, 

              'objective': 'regression', 'reg_alpha': 0.8, 

              'reg_lambda': 0.8, 'subsample': 0.5 }



lgtrain = lightgbm.Dataset(X_train, label=y_train)

lgval = lightgbm.Dataset(X_valid, label=y_valid)



model_lgb = lightgbm.train(

    params, lgtrain, boosting_rounds, valid_sets=[lgtrain, lgval], 

    early_stopping_rounds=1000, verbose_eval=500)



# evaluate using validation set

evals = model_lgb.predict(X_valid)

lmae = competition_metric(valid_df, evals, verbose=1)

print("Log of MAE = {}".format(lmae))



del valid_df, X_train, y_train, X_valid, y_valid

gc.collect()
# predict for test set

X_test = test[selected_features].values

preds = model_lgb.predict(X_test)
# save predictions

test["scalar_coupling_constant"] = preds

test[["id", "scalar_coupling_constant"]].to_csv(f"{download_dir}preds.csv", index=False)