import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import random

import os

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import StandardScaler

lbl = LabelEncoder()

color = sns.color_palette()

sns.set_style('darkgrid')

%matplotlib inline
path = '../input/champs-scalar-coupling/'
train = pd.read_csv(path+'/train.csv')

test = pd.read_csv(path+'test.csv')
train.head()
train.shape
plt.figure(figsize=(8,6))

sns.countplot(train['type'])

plt.show()
plt.figure(figsize=(8,6))

sns.countplot(train['atom_index_0'])

plt.show()
plt.figure(figsize=(8,6))

sns.countplot(train['atom_index_1'])

plt.show()
train = train.sample(frac=0.09, random_state=5)
test = pd.read_csv(path+'/test.csv')
potential_energy = pd.read_csv(path+'/potential_energy.csv')
train.head(5)
pd.isnull(train).sum()
train['type'].unique()
train[['type','scalar_coupling_constant']].groupby(['type'], as_index=False ).mean().sort_values(by='scalar_coupling_constant', ascending=True)
train[['atom_index_0','scalar_coupling_constant']].groupby(['atom_index_0'], as_index=False ).mean().sort_values(by='scalar_coupling_constant', ascending=True)
train[['atom_index_1','scalar_coupling_constant']].groupby(['atom_index_1'], as_index=False ).mean().sort_values(by='scalar_coupling_constant', ascending=True)
for col in train['type'].unique():

    sns.distplot(train[train['type'] == col]['scalar_coupling_constant'])

    plt.show()

    #print(col)
potential_energy.head(5)
potential_energy.isnull().sum()
train  = pd.merge(train, potential_energy, how='left', on='molecule_name', right_index=False)

test  = pd.merge(test, potential_energy, how='left', on='molecule_name', right_index=False )
#test  = pd.merge(test, potential_energy, how='left', on='molecule_name')
train.head(5)
train = train[['id','molecule_name','atom_index_0','atom_index_1','type','potential_energy','scalar_coupling_constant']]
structures = pd.read_csv(path+'/structures.csv')
train.columns
train['atom1'] = train['type'].str[2]

test['atom1'] = test['type'].str[2]
train['atom2'] = train['type'].str[3]

test['atom2'] = test['type'].str[3]
train['coupling_type'] = train['type'].str[0:2]

test['coupling_type'] = test['type'].str[0:2]
train.isnull().sum()
structures.rename(columns={'x':'x1','y':'y1','z':'z1'}, inplace=True)

structures.head()
train = pd.merge(train, structures, how ='left', left_on=['molecule_name', 'atom_index_0','atom1'], right_on=['molecule_name', 'atom_index','atom'], right_index=False)

test = pd.merge(test, structures, how ='left', left_on=['molecule_name', 'atom_index_0','atom1'], right_on=['molecule_name', 'atom_index','atom'], right_index=False)
structures.rename(columns={'x1':'x2','y1':'y2','z1':'z2'}, inplace=True)
train = pd.merge(train, structures, how ='left', left_on=['molecule_name', 'atom_index_1','atom2'], right_on=['molecule_name', 'atom_index','atom'], right_index=False)

test = pd.merge(test, structures, how ='left', left_on=['molecule_name', 'atom_index_1','atom2'], right_on=['molecule_name', 'atom_index','atom'], right_index=False)
train['x2-x1'] = train['x2']-train['x1']

train['y2-y1'] = train['y2']-train['y1']

train['z2-z1'] = train['z2']-train['z1']



test['x2-x1'] = test['x2']-test['x1']

test['y2-y1'] = test['y2']-test['y1']

test['z2-z1'] = test['z2']-test['z1']
train['pow(x2-x1)'] = train['x2-x1']**2

train['pow(y2-y1)'] = train['y2-y1']**2

train['pow(z2-z1)'] = train['z2-z1']**2



test['pow(x2-x1)'] = test['x2-x1']**2

test['pow(y2-y1)'] = test['y2-y1']**2

test['pow(z2-z1)'] = test['z2-z1']**2
train['pow(x2-x1)+pow(y2-y1)+pow(z2-z1)'] = train['pow(x2-x1)']+train['pow(y2-y1)']+train['pow(z2-z1)']



test['pow(x2-x1)+pow(y2-y1)+pow(z2-z1)'] = test['pow(x2-x1)']+test['pow(y2-y1)']+test['pow(z2-z1)']
train['distance'] = np.sqrt(train['pow(x2-x1)+pow(y2-y1)+pow(z2-z1)'])



test['distance'] = np.sqrt(test['pow(x2-x1)+pow(y2-y1)+pow(z2-z1)'])
train.columns
print(os.listdir(path))
mulliken_charges = pd.read_csv(path+'/mulliken_charges.csv')
mulliken_charges.head()
train = pd.merge(train, mulliken_charges, how='left', left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name','atom_index'], right_index=False)



test = pd.merge(test, mulliken_charges, how='left', left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name','atom_index'], right_index=False)
train.rename(columns ={'mulliken_charge':'mulliken_charge_0'}, inplace=True)
train = pd.merge(train, mulliken_charges, how='left', left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name','atom_index'], right_index=False)



test = pd.merge(test, mulliken_charges, how='left', left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name','atom_index'], right_index=False)
train.rename(columns ={'mulliken_charge':'mulliken_charge_1'}, inplace=True)
sc_contributions = pd.read_csv(path+'/scalar_coupling_contributions.csv')
train = pd.merge(train, sc_contributions, how='left', left_on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'], right_on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'], right_index=False)



test = pd.merge(test, sc_contributions, how='left', left_on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'], right_on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'], right_index=False)
dipole_moments = pd.read_csv(path+'/dipole_moments.csv')
dipole_moments.head()
train = pd.merge(train, dipole_moments, how='left', left_on=['molecule_name'], right_on=['molecule_name'], right_index=False)



test = pd.merge(test, dipole_moments, how='left', left_on=['molecule_name'], right_on=['molecule_name'], right_index=False)
train.head()
ms_tensors = pd.read_csv(path+'/magnetic_shielding_tensors.csv')
ms_tensors.head()
train[['atom1','scalar_coupling_constant']].groupby(['atom1'], as_index=False ).mean().sort_values(by='scalar_coupling_constant', ascending=True)
train[['atom2','scalar_coupling_constant']].groupby(['atom2'], as_index=False ).mean().sort_values(by='scalar_coupling_constant', ascending=True)
train[['coupling_type','scalar_coupling_constant']].groupby(['coupling_type'], as_index=False ).mean().sort_values(by='scalar_coupling_constant', ascending=True)
train1 = train.copy()
train1['atom1'] = lbl.fit_transform(train1['atom1'])



test['atom1'] = lbl.fit_transform(test['atom1'])
train1['atom2'] = lbl.fit_transform(train1['atom2'])



test['atom2'] = lbl.fit_transform(test['atom2'])
train1['coupling_type'] = lbl.fit_transform(train1['coupling_type'])



test['coupling_type'] = lbl.fit_transform(test['coupling_type'])
train1['potential_energy'] = lbl.fit_transform(train1['potential_energy'])



test['potential_energy'] = lbl.fit_transform(test['potential_energy'])
#train1['scalar_coupling_constant'] = lbl.fit_transform(train1['scalar_coupling_constant'])
train1.head()
X = np.array(train1[['atom_index_0', 'atom_index_1', 'atom1','atom2', 'coupling_type', 'potential_energy']])

Y = np.array(train1['scalar_coupling_constant'])
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.999900, random_state=52)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.svm import SVC, LinearSVC

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
model_name = []

model_score =[]
'''linsvc = LinearSVC()

linsvc.fit(X_train,Y_train)

linsvc_score = round(linsvc.score(X_train,Y_train)*100, 2)

model_name.append('LinearSVC')

model_score.append(linsvc_score)

linsvc_score'''

'''svc = SVC()

svc.fit(X_train,Y_train)

svc_score = round(svc.score(X_train,Y_train)*100, 2)

model_name.append('SVC')

model_score.append(svc_score)

svc_score'''
'''kneighbors = KNeighborsClassifier()

kneighbors.fit(X_train,Y_train)

kneighbors_score = round(kneighbors.score(X_train,Y_train)*100, 2)

model_name.append('KNeighborsClassifier')

model_score.append(kneighbors_score)

kneighbors_score'''
randomforest = RandomForestRegressor()

randomforest.fit(X_train,Y_train)

randomforest_score = round(randomforest.score(X_train,Y_train)*100, 2)

model_name.append('RandomForestRegressor')

model_score.append(randomforest_score)

randomforest_score
gradient = GradientBoostingRegressor()

gradient.fit(X_train,Y_train)

gradient_score = round(gradient.score(X_train,Y_train)*100, 2)

model_name.append('GradientBoostingRegressor')

model_score.append(gradient_score)

gradient_score
all_score = pd.DataFrame({'model_name':model_name, 'model_score':model_score})

all_score
selected_col = ['atom_index_0', 'atom_index_1', 'atom1','atom2', 'coupling_type', 'potential_energy']



predict_result = randomforest.predict(test[selected_col])
submission = pd.DataFrame({'id':test['id'], 'scalar_coupling_constant':predict_result})

submission.to_csv('my_submission.csv', index=False)