import pandas as pd

import numpy as np



from sklearn.preprocessing import LabelEncoder



import xgboost as xgb



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')

structures = pd.read_csv('../input/structures.csv')
structures = pd.read_csv('../input/structures.csv')



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



train = map_atom_info(train, 0)

train = map_atom_info(train, 1)



test = map_atom_info(test, 0)

test = map_atom_info(test, 1)
train_p_0 = train[['x_0', 'y_0', 'z_0']].values

train_p_1 = train[['x_1', 'y_1', 'z_1']].values

test_p_0 = test[['x_0', 'y_0', 'z_0']].values

test_p_1 = test[['x_1', 'y_1', 'z_1']].values



train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)

test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)

train['dist_x'] = (train['x_0'] - train['x_1']) ** 2

test['dist_x'] = (test['x_0'] - test['x_1']) ** 2

train['dist_y'] = (train['y_0'] - train['y_1']) ** 2

test['dist_y'] = (test['y_0'] - test['y_1']) ** 2

train['dist_z'] = (train['z_0'] - train['z_1']) ** 2

test['dist_z'] = (test['z_0'] - test['z_1']) ** 2
train['type_0'] = train['type'].apply(lambda x: x[0])

test['type_0'] = test['type'].apply(lambda x: x[0])
def create_features(df):

    df['molecule_couples'] = df.groupby('molecule_name')['id'].transform('count')

    df['molecule_dist_mean'] = df.groupby('molecule_name')['dist'].transform('mean')

    df['molecule_dist_min'] = df.groupby('molecule_name')['dist'].transform('min')

    df['molecule_dist_max'] = df.groupby('molecule_name')['dist'].transform('max')

    df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')

    df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')

    df[f'molecule_atom_index_0_x_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('std')

    df[f'molecule_atom_index_0_y_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('mean')

    df[f'molecule_atom_index_0_y_1_mean_diff'] = df[f'molecule_atom_index_0_y_1_mean'] - df['y_1']

    df[f'molecule_atom_index_0_y_1_mean_div'] = df[f'molecule_atom_index_0_y_1_mean'] / df['y_1']

    df[f'molecule_atom_index_0_y_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('max')

    df[f'molecule_atom_index_0_y_1_max_diff'] = df[f'molecule_atom_index_0_y_1_max'] - df['y_1']

    df[f'molecule_atom_index_0_y_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('std')

    df[f'molecule_atom_index_0_z_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('std')

    df[f'molecule_atom_index_0_dist_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('mean')

    df[f'molecule_atom_index_0_dist_mean_diff'] = df[f'molecule_atom_index_0_dist_mean'] - df['dist']

    df[f'molecule_atom_index_0_dist_mean_div'] = df[f'molecule_atom_index_0_dist_mean'] / df['dist']

    df[f'molecule_atom_index_0_dist_max'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('max')

    df[f'molecule_atom_index_0_dist_max_diff'] = df[f'molecule_atom_index_0_dist_max'] - df['dist']

    df[f'molecule_atom_index_0_dist_max_div'] = df[f'molecule_atom_index_0_dist_max'] / df['dist']

    df[f'molecule_atom_index_0_dist_min'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')

    df[f'molecule_atom_index_0_dist_min_diff'] = df[f'molecule_atom_index_0_dist_min'] - df['dist']

    df[f'molecule_atom_index_0_dist_min_div'] = df[f'molecule_atom_index_0_dist_min'] / df['dist']

    df[f'molecule_atom_index_0_dist_std'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('std')

    df[f'molecule_atom_index_0_dist_std_diff'] = df[f'molecule_atom_index_0_dist_std'] - df['dist']

    df[f'molecule_atom_index_0_dist_std_div'] = df[f'molecule_atom_index_0_dist_std'] / df['dist']

    df[f'molecule_atom_index_1_dist_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('mean')

    df[f'molecule_atom_index_1_dist_mean_diff'] = df[f'molecule_atom_index_1_dist_mean'] - df['dist']

    df[f'molecule_atom_index_1_dist_mean_div'] = df[f'molecule_atom_index_1_dist_mean'] / df['dist']

    df[f'molecule_atom_index_1_dist_max'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('max')

    df[f'molecule_atom_index_1_dist_max_diff'] = df[f'molecule_atom_index_1_dist_max'] - df['dist']

    df[f'molecule_atom_index_1_dist_max_div'] = df[f'molecule_atom_index_1_dist_max'] / df['dist']

    df[f'molecule_atom_index_1_dist_min'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('min')

    df[f'molecule_atom_index_1_dist_min_diff'] = df[f'molecule_atom_index_1_dist_min'] - df['dist']

    df[f'molecule_atom_index_1_dist_min_div'] = df[f'molecule_atom_index_1_dist_min'] / df['dist']

    df[f'molecule_atom_index_1_dist_std'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('std')

    df[f'molecule_atom_index_1_dist_std_diff'] = df[f'molecule_atom_index_1_dist_std'] - df['dist']

    df[f'molecule_atom_index_1_dist_std_div'] = df[f'molecule_atom_index_1_dist_std'] / df['dist']

    df[f'molecule_atom_1_dist_mean'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('mean')

    df[f'molecule_atom_1_dist_min'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('min')

    df[f'molecule_atom_1_dist_min_diff'] = df[f'molecule_atom_1_dist_min'] - df['dist']

    df[f'molecule_atom_1_dist_min_div'] = df[f'molecule_atom_1_dist_min'] / df['dist']

    df[f'molecule_atom_1_dist_std'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('std')

    df[f'molecule_atom_1_dist_std_diff'] = df[f'molecule_atom_1_dist_std'] - df['dist']

    df[f'molecule_type_0_dist_std'] = df.groupby(['molecule_name', 'type_0'])['dist'].transform('std')

    df[f'molecule_type_0_dist_std_diff'] = df[f'molecule_type_0_dist_std'] - df['dist']

    df[f'molecule_type_dist_mean'] = df.groupby(['molecule_name', 'type'])['dist'].transform('mean')

    df[f'molecule_type_dist_mean_diff'] = df[f'molecule_type_dist_mean'] - df['dist']

    df[f'molecule_type_dist_mean_div'] = df[f'molecule_type_dist_mean'] / df['dist']

    df[f'molecule_type_dist_max'] = df.groupby(['molecule_name', 'type'])['dist'].transform('max')

    df[f'molecule_type_dist_min'] = df.groupby(['molecule_name', 'type'])['dist'].transform('min')

    df[f'molecule_type_dist_std'] = df.groupby(['molecule_name', 'type'])['dist'].transform('std')

    df[f'molecule_type_dist_std_diff'] = df[f'molecule_type_dist_std'] - df['dist']



    return df
good_columns = [

'molecule_atom_index_0_dist_min',

'molecule_atom_index_0_dist_max',

'molecule_atom_index_1_dist_min',

'molecule_atom_index_0_dist_mean',

'molecule_atom_index_0_dist_std',

'dist',

'molecule_atom_index_1_dist_std',

'molecule_atom_index_1_dist_max',

'molecule_atom_index_1_dist_mean',

'molecule_atom_index_0_dist_max_diff',

'molecule_atom_index_0_dist_max_div',

'molecule_atom_index_0_dist_std_diff',

'molecule_atom_index_0_dist_std_div',

'atom_0_couples_count',

'molecule_atom_index_0_dist_min_div',

'molecule_atom_index_1_dist_std_diff',

'molecule_atom_index_0_dist_mean_div',

'atom_1_couples_count',

'molecule_atom_index_0_dist_mean_diff',

'molecule_couples',

'atom_index_1',

'molecule_dist_mean',

'molecule_atom_index_1_dist_max_diff',

'molecule_atom_index_0_y_1_std',

'molecule_atom_index_1_dist_mean_diff',

'molecule_atom_index_1_dist_std_div',

'molecule_atom_index_1_dist_mean_div',

'molecule_atom_index_1_dist_min_diff',

'molecule_atom_index_1_dist_min_div',

'molecule_atom_index_1_dist_max_div',

'molecule_atom_index_0_z_1_std',

'y_0',

'molecule_type_dist_std_diff',

'molecule_atom_1_dist_min_diff',

'molecule_atom_index_0_x_1_std',

'molecule_dist_min',

'molecule_atom_index_0_dist_min_diff',

'molecule_atom_index_0_y_1_mean_diff',

'molecule_type_dist_min',

'molecule_atom_1_dist_min_div',

'atom_index_0',

'molecule_dist_max',

'molecule_atom_1_dist_std_diff',

'molecule_type_dist_max',

'molecule_atom_index_0_y_1_max_diff',

'molecule_type_0_dist_std_diff',

'molecule_type_dist_mean_diff',

'molecule_atom_1_dist_mean',

'molecule_atom_index_0_y_1_mean_div',

'molecule_type_dist_mean_div',

'type'

]
train = create_features(train)

test = create_features(test)
for f in ['atom_index_0', 'atom_index_1', 'atom_1', 'type_0', 'type']:

    if f in good_columns:

        lbl = LabelEncoder()

        lbl.fit(list(train[f].values) + list(test[f].values))

        train[f] = lbl.transform(list(train[f].values))

        test[f] = lbl.transform(list(test[f].values))
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
good_train_columns = [

'molecule_atom_index_0_dist_min',

'molecule_atom_index_0_dist_max',

'molecule_atom_index_1_dist_min',

'molecule_atom_index_0_dist_mean',

'molecule_atom_index_0_dist_std',

'dist',

'molecule_atom_index_1_dist_std',

'molecule_atom_index_1_dist_max',

'molecule_atom_index_1_dist_mean',

'molecule_atom_index_0_dist_max_diff',

'molecule_atom_index_0_dist_max_div',

'molecule_atom_index_0_dist_std_diff',

'molecule_atom_index_0_dist_std_div',

'atom_0_couples_count',

'molecule_atom_index_0_dist_min_div',

'molecule_atom_index_1_dist_std_diff',

'molecule_atom_index_0_dist_mean_div',

'atom_1_couples_count',

'molecule_atom_index_0_dist_mean_diff',

'molecule_couples',

'atom_index_1',

'molecule_dist_mean',

'molecule_atom_index_1_dist_max_diff',

'molecule_atom_index_0_y_1_std',

'molecule_atom_index_1_dist_mean_diff',

'molecule_atom_index_1_dist_std_div',

'molecule_atom_index_1_dist_mean_div',

'molecule_atom_index_1_dist_min_diff',

'molecule_atom_index_1_dist_min_div',

'molecule_atom_index_1_dist_max_div',

'molecule_atom_index_0_z_1_std',

'y_0',

'molecule_type_dist_std_diff',

'molecule_atom_1_dist_min_diff',

'molecule_atom_index_0_x_1_std',

'molecule_dist_min',

'molecule_atom_index_0_dist_min_diff',

'molecule_atom_index_0_y_1_mean_diff',

'molecule_type_dist_min',

'molecule_atom_1_dist_min_div',

'atom_index_0',

'molecule_dist_max',

'molecule_atom_1_dist_std_diff',

'molecule_type_dist_max',

'molecule_atom_index_0_y_1_max_diff',

'molecule_type_0_dist_std_diff',

'molecule_type_dist_mean_diff',

'molecule_atom_1_dist_mean',

'molecule_atom_index_0_y_1_mean_div',

'molecule_type_dist_mean_div',

'type',

'scalar_coupling_constant'

]
train = train[good_train_columns]

train = reduce_mem_usage(train)

length = len(train)

train1 = train[0:int(length*(1/4))]

train2 = train[int(length*(1/4)):int(length*(2/4))]

train3 = train[int(length*(2/4)):int(length*(3/4))]

train4 = train[int(length*(3/4)):int(length*(4/4))]



del train
good_test_columns = [

'molecule_atom_index_0_dist_min',

'molecule_atom_index_0_dist_max',

'molecule_atom_index_1_dist_min',

'molecule_atom_index_0_dist_mean',

'molecule_atom_index_0_dist_std',

'dist',

'molecule_atom_index_1_dist_std',

'molecule_atom_index_1_dist_max',

'molecule_atom_index_1_dist_mean',

'molecule_atom_index_0_dist_max_diff',

'molecule_atom_index_0_dist_max_div',

'molecule_atom_index_0_dist_std_diff',

'molecule_atom_index_0_dist_std_div',

'atom_0_couples_count',

'molecule_atom_index_0_dist_min_div',

'molecule_atom_index_1_dist_std_diff',

'molecule_atom_index_0_dist_mean_div',

'atom_1_couples_count',

'molecule_atom_index_0_dist_mean_diff',

'molecule_couples',

'atom_index_1',

'molecule_dist_mean',

'molecule_atom_index_1_dist_max_diff',

'molecule_atom_index_0_y_1_std',

'molecule_atom_index_1_dist_mean_diff',

'molecule_atom_index_1_dist_std_div',

'molecule_atom_index_1_dist_mean_div',

'molecule_atom_index_1_dist_min_diff',

'molecule_atom_index_1_dist_min_div',

'molecule_atom_index_1_dist_max_div',

'molecule_atom_index_0_z_1_std',

'y_0',

'molecule_type_dist_std_diff',

'molecule_atom_1_dist_min_diff',

'molecule_atom_index_0_x_1_std',

'molecule_dist_min',

'molecule_atom_index_0_dist_min_diff',

'molecule_atom_index_0_y_1_mean_diff',

'molecule_type_dist_min',

'molecule_atom_1_dist_min_div',

'atom_index_0',

'molecule_dist_max',

'molecule_atom_1_dist_std_diff',

'molecule_type_dist_max',

'molecule_atom_index_0_y_1_max_diff',

'molecule_type_0_dist_std_diff',

'molecule_type_dist_mean_diff',

'molecule_atom_1_dist_mean',

'molecule_atom_index_0_y_1_mean_div',

'molecule_type_dist_mean_div',

'type'

]
test = test[good_test_columns]

test = reduce_mem_usage(test)

X_test = test



del test
X_train1 = train1.drop(['scalar_coupling_constant'], axis=1)

y_train1 = train1['scalar_coupling_constant']

X_train2 = train2.drop(['scalar_coupling_constant'], axis=1)

y_train2 = train2['scalar_coupling_constant']

X_train3 = train3.drop(['scalar_coupling_constant'], axis=1)

y_train3 = train3['scalar_coupling_constant']

X_train4 = train4.drop(['scalar_coupling_constant'], axis=1)

y_train4 = train4['scalar_coupling_constant']



del train1, train2, train3, train4
model = xgb.XGBRegressor(n_jobs=-1, random_state=42)

model.fit(X_train1, y_train1)

model.fit(X_train2, y_train2)

model.fit(X_train3, y_train3)

model.fit(X_train4, y_train4)
pred = model.predict(X_test)

sub =pd.read_csv('../input/sample_submission.csv')

sub['scalar_coupling_constant'] = pred

sub.to_csv('../submission.csv', index=False)