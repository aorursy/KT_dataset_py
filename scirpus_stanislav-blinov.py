import gc

import pandas as pd

import numpy as np

from scipy.sparse import csr_matrix

from sklearn.preprocessing import LabelEncoder, StandardScaler

from scipy.sparse import csr_matrix

from sklearn.decomposition import TruncatedSVD

import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

from sklearn import metrics

# Graph Stuff

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df = pd.read_csv('../input/train.csv')

df.atom_index_0+=1

df.atom_index_1+=1
df.molecule_name.value_counts().tail()
trainmolecules = df.molecule_name.unique()

le = LabelEncoder()

df.type= le.fit_transform(df.type)+1

df.head()
x = df.atom_index_0-df.atom_index_1

print(x.min())

print(x.max())
df_structures = pd.read_csv('../input/structures.csv')

df_structures.atom_index+=1

df_structures.head(20)
le = LabelEncoder()

df_structures.atom = le.fit_transform(df_structures.atom)+1
df_structures.head(20)
z = None

for i in np.arange(0,5,1):

    print(i)

    if(z is None):

        z = df[['molecule_name','atom_index_0','atom_index_1','type','scalar_coupling_constant']].copy()

    y = df_structures.copy()

    y.atom_index -= i

   

    z = z.merge(y, how='left',left_on=['molecule_name','atom_index_0'], right_on=['molecule_name','atom_index'],suffixes=('', '_neg_'+str(i)))

    if(z.columns.contains('atom_index_neg_'+str(i))):

        del z['atom_index_neg_'+str(i)]



    if(i!=0):

        z['dist_neg_'+str(i)] = 1./np.linalg.norm(z[['x', 'y', 'z']].values - z[['x_neg_'+str(i), 'y_neg_'+str(i), 'z_neg_'+str(i)]].values, axis=1)

        del z['x_neg_'+str(i)]

        del z['y_neg_'+str(i)]

        del z['z_neg_'+str(i)]



    

for i in np.arange(1,5,1):

    print(i)

    

    y = df_structures.copy()

    y.atom_index += i



    

    z = z.merge(y, how='left',left_on=['molecule_name','atom_index_0'], right_on=['molecule_name','atom_index'],suffixes=('', '_pos_'+str(i)))    

    if(z.columns.contains('atom_index')):

        del z['atom_index']

    if(z.columns.contains('atom_index_pos_'+str(i))):

        del z['atom_index_pos_'+str(i)]

    z['dist_pos_'+str(i)] = 1./np.linalg.norm(z[['x', 'y', 'z']].values - z[['x_pos_'+str(i), 'y_pos_'+str(i), 'z_pos_'+str(i)]].values, axis=1)

    del z['x_pos_'+str(i)]

    del z['y_pos_'+str(i)]

    del z['z_pos_'+str(i)]

del z['x']

del z['y']

del z['z']    



for i in np.arange(0,5,1):

    print(i)

    y = df_structures.copy()

    y.atom_index -= i

   

    z = z.merge(y, how='left',left_on=['molecule_name','atom_index_1'], right_on=['molecule_name','atom_index'],suffixes=('', '_1_neg_'+str(i)))

    if(z.columns.contains('atom_index_1_neg_'+str(i))):

        del z['atom_index_1_neg_'+str(i)]



    if(i!=0):

        z['dist_1_neg_'+str(i)] = 1./np.linalg.norm(z[['x', 'y', 'z']].values - z[['x_1_neg_'+str(i), 'y_1_neg_'+str(i), 'z_1_neg_'+str(i)]].values, axis=1)

        del z['x_1_neg_'+str(i)]

        del z['y_1_neg_'+str(i)]

        del z['z_1_neg_'+str(i)]



    

for i in np.arange(1,5,1):

    print(i)



    y = df_structures.copy()

    y.atom_index += i

   

    z = z.merge(y, how='left',left_on=['molecule_name','atom_index_1'], right_on=['molecule_name','atom_index'],suffixes=('', '_1_pos_'+str(i)))    

    if(z.columns.contains('atom_index')):

        del z['atom_index']

    if(z.columns.contains('atom_index_1_pos_'+str(i))):

        del z['atom_index_1_pos_'+str(i)]

    z['dist_1_pos_'+str(i)] = 1./np.linalg.norm(z[['x', 'y', 'z']].values - z[['x_1_pos_'+str(i), 'y_1_pos_'+str(i), 'z_1_pos_'+str(i)]].values, axis=1)

    del z['x_1_pos_'+str(i)]

    del z['y_1_pos_'+str(i)]

    del z['z_1_pos_'+str(i)]    

  

del z['x']

del z['y']

del z['z']
df.shape
z.shape
z.dropna(axis = 1, how ='all', inplace = True)

sc = z.scalar_coupling_constant.values

del z['scalar_coupling_constant']

z['scalar_coupling_constant'] = sc
def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):

    """

    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling

    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric

    """

    maes = (y_true-y_pred).abs().groupby(types).mean()

    return np.log(maes.map(lambda x: max(x, floor))).mean()
n_fold = 5

folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
df = z
feats = df.columns[1:-1]

X = csr_matrix(df[feats].fillna(0).values)

y = df.scalar_coupling_constant.values
params = {'num_leaves': 128,

          'min_child_samples': 79,

          'objective': 'regression',

          'max_depth': -1,

          'learning_rate': 0.2,

          "boosting_type": "gbdt",

          "subsample_freq": 1,

          "subsample": 0.9,

          "bagging_seed": 11,

          "metric": 'mae',

          "verbosity": -1,

          'reg_alpha': 0.1,

          'reg_lambda': 0.3,

          'colsample_bytree': 1.0

          

         }



# out-of-fold predictions on train data

oof = np.zeros(X.shape[0])

for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

    print(fold_n)

    X_train, X_valid = X[train_index], X[valid_index]

    y_train, y_valid = y[train_index], y[valid_index]

    model = lgb.LGBMRegressor(**params, n_estimators = 1500)

    model.fit(X_train, y_train, 

            eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',

            verbose=500, early_stopping_rounds=200)



    y_pred_valid = model.predict(X_valid)

    oof[valid_index] = y_pred_valid.reshape(-1,)

    break
x = pd.DataFrame()

x['target'] = y

x['predictions'] = oof

x['type'] = df.type.values

x = x[x.predictions!=0]



group_mean_log_mae(x.target, x.predictions, x.type, floor=1e-9)
plot_data = pd.DataFrame()

plot_data['yhat'] = x['predictions'].values

plot_data['y'] = x['target'].values

plot_data['type'] = x['type'].values
for t in sorted([1, 4, 2, 5, 3, 7, 6, 8]):

    plt.scatter(plot_data[plot_data.type==t].y,

                plot_data[plot_data.type==t].yhat,s=1)

    plt.show()