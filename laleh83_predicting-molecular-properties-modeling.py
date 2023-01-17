#import libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# load the data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')



# get the shape of the train and test sets

train_shape = train.shape

test_shape = test.shape



print('Train shape: {}'.format(train_shape))

print('Test shape: {}'.format(test_shape))
sample_submission.head()
train.head()
test.head()
structures = pd.read_csv('../input/structures.csv')

dipole_moments = pd.read_csv('../input/dipole_moments.csv')

magnetic_shielding_tensors = pd.read_csv('../input/magnetic_shielding_tensors.csv')

mulliken_charges = pd.read_csv('../input/mulliken_charges.csv')

potential_energy = pd.read_csv('../input/potential_energy.csv')

scalar_coupling_contributions = pd.read_csv('../input/scalar_coupling_contributions.csv')
dipole_moments.head()
magnetic_shielding_tensors.head()
mulliken_charges.head()
potential_energy.head()
scalar_coupling_contributions.head()
train.info()
test.info()
train.describe()
test.describe()
train['type'].unique()
test.columns
# plot the categorical features - type

train.groupby("type").id.count().sort_values(ascending=False)[:10].plot.bar()
# plot the distribution of the median of target variable by type

import matplotlib.pyplot as plt

plt.style.use('ggplot')



molecule_type = train.groupby('type',as_index=False)['scalar_coupling_constant'].median()

fig = plt.figure(figsize=(7,5))

plt.bar(molecule_type.type,molecule_type.scalar_coupling_constant,width=0.5,alpha=0.8)



plt.xlabel('scalar_coupling_constant')

plt.ylabel('Median Scalar Constant')

plt.title('Median of scalar constant across type')

plt.show()
import seaborn as sns

sns.distplot(train['scalar_coupling_constant'])
#skewness and kurtosis of the target variable

print("Skewness: ", train['scalar_coupling_constant'].skew())

print("Kurtosis: ", train['scalar_coupling_constant'].kurt())
train.scalar_coupling_constant.unique()
# re-load the train and test sets in their raw state



# train = pd.read_csv('../input/train.csv')

# test = pd.read_csv('../input/test.csv')
df_all = pd.concat([train,test],sort=False)
# create new features using for joining main data to the structures data



df_all['jcoupling_type'] = df_all['type'].str.extract("(\d)J\w") 

df_all['jcoupling_atoms'] = df_all['type'].str.extract("\dJ(\w*)") 

df_all['jcoupling_atom_0'] = df_all.type.str[2]

df_all['jcoupling_atom_1'] = df_all.type.str[3]

df_all['coupling_atom_type'] = df_all['type'].str.extract("\dJ(\w*)") 

df_all['jcoupling_atom_0'] = df_all['atom_index_0'].astype(str) + '_' + df_all['jcoupling_atom_0']

df_all['jcoupling_atom_1'] = df_all['atom_index_1'].astype(str) + '_' + df_all['jcoupling_atom_1']

df_all['jcoupling_atoms'] = df_all['jcoupling_atom_0'].astype(str) + '_' + df_all['jcoupling_atom_1']



structures['jcoupling_atoms'] = structures['atom_index'].astype(str) + '_' + structures['atom']
df_all.head()
df_all.shape # 7,163,689
# join the atom structure xyz to the molecule data



df_new = pd.merge(df_all,structures[['molecule_name','jcoupling_atoms','x','y','z']],left_on=['molecule_name','jcoupling_atom_0'],right_on=['molecule_name','jcoupling_atoms'],how='left')

df_new.drop(columns='jcoupling_atoms_y',inplace=True)

df_new.rename(columns={"x": "x_atom_0", "y": "y_atom_0", "z": "z_atom_0","jcoupling_atoms_x": "jcoupling_atoms"},inplace=True)

df_new.head()
df_new = pd.merge(df_new,structures[['molecule_name','jcoupling_atoms','x','y','z']],left_on=['molecule_name','jcoupling_atom_1'],right_on=['molecule_name','jcoupling_atoms'],how='left')

df_new.drop(columns='jcoupling_atoms_y',inplace=True)

df_new.rename(columns={"x": "x_atom_1", "y": "y_atom_1", "z": "z_atom_1","jcoupling_atoms_x": "jcoupling_atoms"},inplace=True)

df_new.head()
# calculate distance between jcoupling atoms in the molecule structure



df_new['dist'] = np.sqrt( (df_new.x_atom_0-df_new.x_atom_1)**2 + (df_new.y_atom_0-df_new.y_atom_1)**2 + (df_new.z_atom_0-df_new.z_atom_1)**2)

df_new.head()
df_new.shape #7163689
final_features = ['id','molecule_name','atom_index_0','atom_index_1','type','jcoupling_type','coupling_atom_type','dist','scalar_coupling_constant']

molecule_atoms = df_new[final_features]
molecule_atoms.head()
# encode categorical features



from sklearn.preprocessing import LabelEncoder



ohe = pd.get_dummies(molecule_atoms['type'],prefix='ohe')

# molecule_atoms.drop('type',axis=1,inplace=True)

molecule_atoms = pd.concat([molecule_atoms,ohe],axis=1)

molecule_atoms.head()
train = molecule_atoms[molecule_atoms.id.isin(train.id)]

test = molecule_atoms[molecule_atoms.id.isin(test.id)]
test.head()
# drop the target variable from the test set

test = test.drop(columns='scalar_coupling_constant')

test.columns
train.columns
# left join the scalar coupling contribution dataframe to the train set

train_new = pd.merge(train,scalar_coupling_contributions,left_on=['molecule_name','atom_index_0','atom_index_1','type'],right_on=['molecule_name','atom_index_0','atom_index_1','type'],how='left')

train_new.head()
# left join the mulliken_charges and potential_energy dataframes to the train set

train_new = pd.merge(train_new,potential_energy,left_on=['molecule_name'],right_on=['molecule_name'],how='left')

train_new.head()
train_new.columns # 4,658,147
# numerical features except the scalar_coupling_constant,sd, pso, dso, potential_energy - one additional feature at a time to predict

num_features = ['atom_index_0', 'atom_index_1', 'jcoupling_type', 'dist',

       'ohe_1JHC', 'ohe_1JHN', 'ohe_2JHC','ohe_2JHH', 'ohe_2JHN', 'ohe_3JHC', 'ohe_3JHH', 'ohe_3JHN']
# scale the numerical features

from sklearn.preprocessing import StandardScaler



test_new = test.copy()

sc = StandardScaler()

train_new[num_features] = sc.fit_transform(train_new[num_features])

test_new[num_features] = sc.transform(test_new[num_features])
test_new.head()
# predict fc in the test

import xgboost as xgb

from sklearn.model_selection import KFold

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor







# kf = KFold(n_splits=5,shuffle=True,random_state=123)



# fold = 0

# fold_metrics = []

# for train_index, test_index in kf.split(train_new):

#     cv_train, cv_test = train_new.iloc[train_index], train_new.iloc[test_index]

#     regressor = GradientBoostingRegressor()

#     regressor.fit(X=cv_train[num_features],y=cv_train['fc'])

#     predictions = regressor.predict(cv_test[num_features])

#     metric = np.log(mean_absolute_error(cv_test['fc'],predictions))

#     fold_metrics.append(metric)

#     print('Fold:{}'.format(fold))

#     print('CV train shape:{}'.format(cv_train.shape))

#     print('Log mean squared error:{}'.format(metric))

#     fold+=1
# Average and overall metric



# mean_score = np.mean(fold_metrics)

# overal_score_minimizing = np.mean(fold_metrics)+np.std(fold_metrics)

# print(mean_score,overal_score_minimizing)



# lr ---> 1.4350236728166157 1.4368374949231812

# gb ---> 0.978808394321879 0.9800942928921375

# rf ---> 1.0059670776324137 1.0071424059922574

# dt ---> 1.1635835592319883 1.1646638382205583

# xgb --> 0.9794824442750851 0.9803075149244272
# predict the fc feature

from sklearn.ensemble import GradientBoostingRegressor



regressor = GradientBoostingRegressor()

regressor.fit(X=train_new[num_features],y=train_new['fc'])

test_new['fc'] = regressor.predict(test_new[num_features])
# test_new.head()

train = train_new.copy()

test = test_new.copy()
# from sklearn.model_selection import GridSearchCV



# param_grid = {

#     'max_depth' : [3, 5, 7],

#     'subsample' : [0.8, 0.9, 1.0]

# }



# regressor = GradientBoostingRegressor()

# grid_search = GridSearchCV(estimator = regressor, param_grid = param_grid, 

#                           cv = 3,  verbose = 2)

# grid_search.fit(train_new[num_features], train_new['fc'])
features = ['atom_index_0', 'atom_index_1', 'dist','fc', 'ohe_1JHC', 'ohe_1JHN', 'ohe_2JHC',

       'ohe_2JHH', 'ohe_2JHN', 'ohe_3JHC', 'ohe_3JHH', 'ohe_3JHN']

# , 'jcoupling_type'
# Feature Scaling

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

train[features] = sc.fit_transform(train[features])

test[features] = sc.transform(test[features])
train[features].head()
# from sklearn.linear_model import LinearRegression



# # fit the model on the train set

# lr = LinearRegression()

# lr.fit(X=train[features],y=train['scalar_coupling_constant'])
# kfold cross-validation for evaluating the model

from sklearn.model_selection import KFold

from sklearn.metrics import mean_absolute_error



kf = KFold(n_splits=5,shuffle=True,random_state=123)



# fold = 0

# fold_metrics = []

# for train_index, test_index in kf.split(train):

#     cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]

#     lr.fit(X=cv_train[features],y=cv_train['scalar_coupling_constant'])

#     predictions = lr.predict(cv_test[features])

#     metric = np.log(mean_absolute_error(cv_test['scalar_coupling_constant'],predictions))

#     fold_metrics.append(metric)

#     print('Fold:{}'.format(fold))

#     print('CV train shape:{}'.format(cv_train.shape))

#     print('Log mean squared error:{}'.format(metric))

#     fold+=1
# Average and overall metric



# mean_score = np.mean(fold_metrics)

# overal_score_minimizing = np.mean(fold_metrics)+np.std(fold_metrics)

# print(mean_score,overal_score_minimizing)
# test['scalar_coupling_constant'] = lr.predict(test[features])
# submission_2 = test[['id','scalar_coupling_constant']]
# submission_2.head()
# submission_2.shape
# submission_2.to_csv('submission_v2.csv',index=False)
# from sklearn.ensemble import RandomForestRegressor



# rf = RandomForestRegressor()



# fold = 0

# fold_metrics = []

# for train_index, test_index in kf.split(train):

#     cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]

#     rf.fit(X=cv_train[features],y=cv_train['scalar_coupling_constant'])

#     predictions = rf.predict(cv_test[features])

#     metric = np.log(mean_absolute_error(cv_test['scalar_coupling_constant'],predictions))

#     fold_metrics.append(metric)

#     print('Fold:{}'.format(fold))

#     print('CV train shape:{}'.format(cv_train.shape))

#     print('Log mean squared error:{}'.format(metric))

#     fold+=1
# mean_score = np.mean(fold_metrics)

# overal_score_minimizing = np.mean(fold_metrics)+np.std(fold_metrics)

# print(mean_score,overal_score_minimizing)
from sklearn.ensemble import GradientBoostingRegressor



# regressor = GradientBoostingRegressor()



# fold = 0

# fold_metrics = []

# for train_index, test_index in kf.split(train):

#     cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]

#     regressor.fit(X=cv_train[features],y=cv_train['scalar_coupling_constant'])

#     predictions = regressor.predict(cv_test[features])

#     metric = np.log(mean_absolute_error(cv_test['scalar_coupling_constant'],predictions))

#     fold_metrics.append(metric)

#     print('Fold:{}'.format(fold))

#     print('CV train shape:{}'.format(cv_train.shape))

#     print('Log mean squared error:{}'.format(metric))

#     fold+=1

"""

Fold:0

CV train shape:(3726517, 22)

Log mean squared error:-1.651942379074779

Fold:1

CV train shape:(3726517, 22)

Log mean squared error:-1.6581039302273266

Fold:2

CV train shape:(3726518, 22)

Log mean squared error:-1.64427536075409

Fold:3

CV train shape:(3726518, 22)

Log mean squared error:-1.6546618867855556

Fold:4

CV train shape:(3726518, 22)

Log mean squared error:-1.6543852532228525

"""
mean_score = np.mean(fold_metrics)

overal_score_minimizing = np.mean(fold_metrics) + np.std(fold_metrics)

print(mean_score,overal_score_minimizing) # -1.6526737620129208 -1.648038319556876
from sklearn.ensemble import GradientBoostingRegressor



gb = GradientBoostingRegressor()

gb.fit(X= train[features],y=train['scalar_coupling_constant'])

test['scalar_coupling_constant'] = gb.predict(test[features])
submission_5 = test[['id','scalar_coupling_constant']]

submission_5.to_csv('submission_v5.csv',index=False)
submission_4.head()
test.head()