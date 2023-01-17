import os

import gc

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.shape)

train.head(10)
test.head(10)
train.isnull().values.any()
test.isnull().values.any()
pp = pd.value_counts(train.dtypes)

pp.plot.bar()

plt.show()
plt.plot(np.sort(train['bandgap_energy_ev'])) # plotting by columns

plt.show()
plt.plot(np.sort(train['formation_energy_ev_natom'])) # plotting by columns

plt.show()
dummy = train[['id', 'bandgap_energy_ev', 'formation_energy_ev_natom']]

dummy['bandgap_energy_ev'] = np.log1p(dummy['bandgap_energy_ev'])

dummy['formation_energy_ev_natom'] = np.log1p(dummy['formation_energy_ev_natom'])

dummy.head()
g = sns.lmplot(x = 'id', y = "bandgap_energy_ev", data = dummy, fit_reg = False)
g = sns.lmplot(x = 'id', y = 'formation_energy_ev_natom', data = dummy, fit_reg = False)
yy = pd.value_counts(train['spacegroup'])



fig, ax = plt.subplots()

fig.set_size_inches(11.7, 8.27)

sns.set_style("whitegrid")



ax = sns.barplot(x=yy.index, y=yy, data=train)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)

ax.set(xlabel='space groups', ylabel='Count')

ax.set_title('Distribution of space groups')
f, ax = plt.subplots(figsize=(12, 7))

ax = sns.countplot(hue = "number_of_total_atoms", y = 'spacegroup', data = train)
df = train.groupby(['spacegroup', 'number_of_total_atoms']).mean()

df
train['formation_energy_ev_natom'] = np.log1p(train.formation_energy_ev_natom)

train['bandgap_energy_ev'] = np.log1p(train.bandgap_energy_ev)



train.head(10)
ccol = []

for i in train.columns:

    if i not in ['id', 'bandgap_energy_ev', 'formation_energy_ev_natom']:

        ccol.append(i)



ccol
train.columns
def new_second_order(df, c_names):

    names_col=[]

    pp=0

    for i in c_names[:len(c_names)-1]:

        for j in c_names[pp:len(c_names)]:

            if i != j:

                col_name = i + str('_*_') + j

                df[col_name] = df[i] * df[j] 

        pp+=1

    return df, names_col   



train, _ = new_second_order(train, ccol)

test, _ = new_second_order(test, ccol)
print(len(train.columns))

print(len(test.columns))
def new_features(df):

    df['percent_atom_al_ga'] = df['percent_atom_al'] * df['percent_atom_ga']

    df['percent_atom_al_in'] = df['percent_atom_al'] * df['percent_atom_in']

    df['percent_atom_ga_in'] = df['percent_atom_ga'] * df['percent_atom_in']

    

    df['lattice_vector_1_ang_/_2'] = df['lattice_vector_1_ang'] / df['lattice_vector_2_ang']

    df['lattice_vector_2_ang_/_3'] = df['lattice_vector_2_ang'] / df['lattice_vector_3_ang']

    df['lattice_vector_3_ang_/_1'] = df['lattice_vector_3_ang'] / df['lattice_vector_1_ang']

    

    df['lattice_angle_alpha_beta_degree'] = df['lattice_angle_alpha_degree'] * df['lattice_angle_beta_degree']

    df['lattice_angle_beta_gamma_degree'] = df['lattice_angle_beta_degree'] * df['lattice_angle_gamma_degree']

    df['lattice_angle_gamma_alpha_degree'] = df['lattice_angle_gamma_degree'] * df['lattice_angle_alpha_degree']



new_features(train)  

new_features(test)
print(len(train.columns))

print(len(test.columns))
#'''

def newer_features(df):

    df['percent_atom_al_ga_in/lv1'] = (df['percent_atom_al'] + df['percent_atom_ga'] + df['percent_atom_in']) / df['lattice_vector_1_ang']

    df['percent_atom_al_ga_in/lv2'] = (df['percent_atom_al'] + df['percent_atom_ga'] + df['percent_atom_in']) / df['lattice_vector_2_ang']

    df['percent_atom_al_ga_in/lv3'] = (df['percent_atom_al'] + df['percent_atom_ga'] + df['percent_atom_in']) / df['lattice_vector_3_ang']

    

    df['percent_atom_al_ga_in*alpha'] = (df['percent_atom_al'] + df['percent_atom_ga'] + df['percent_atom_in']) * df['lattice_angle_alpha_degree']

    df['percent_atom_al_ga_in*beta'] = (df['percent_atom_al'] + df['percent_atom_ga'] + df['percent_atom_in']) * df['lattice_angle_beta_degree']

    df['percent_atom_al_ga_in*gamma'] = (df['percent_atom_al'] + df['percent_atom_ga'] + df['percent_atom_in']) * df['lattice_angle_gamma_degree']

    

    df['lattice_vector_A_B_G_1'] = np.sqrt(df['lattice_angle_alpha_degree'] * df['lattice_angle_beta_degree'] * df['lattice_angle_gamma_degree']) / df['lattice_vector_1_ang']

    df['lattice_vector_A_B_G_2'] = np.sqrt(df['lattice_angle_alpha_degree'] * df['lattice_angle_beta_degree'] * df['lattice_angle_gamma_degree']) / df['lattice_vector_2_ang']

    df['lattice_vector_A_B_G_3'] = np.sqrt(df['lattice_angle_alpha_degree'] * df['lattice_angle_beta_degree'] * df['lattice_angle_gamma_degree']) / df['lattice_vector_3_ang']

    

    df['lattice_123_A_B'] = (df['lattice_vector_1_ang'] + df['lattice_vector_2_ang'] + df['lattice_vector_3_ang']) / (df['lattice_angle_alpha_degree'] * df['lattice_angle_beta_degree'])

    df['lattice_123_B_G'] = (df['lattice_vector_1_ang'] + df['lattice_vector_2_ang'] + df['lattice_vector_3_ang']) / (df['lattice_angle_beta_degree'] * df['lattice_angle_gamma_degree'])

    df['lattice_123_G_A'] = (df['lattice_vector_1_ang'] + df['lattice_vector_2_ang'] + df['lattice_vector_3_ang']) / (df['lattice_angle_gamma_degree'] * df['lattice_angle_alpha_degree'])



newer_features(train)  

newer_features(test)

#'''
print(len(train.columns))

print(len(test.columns))
def new_feat(df):

    df['NTA_al_ga_in'] = df['number_of_total_atoms'] * (df['percent_atom_al'] + df['percent_atom_ga'] + df['percent_atom_in'])

    df['NTA_1_2_3'] = df['number_of_total_atoms'] * (df['lattice_vector_1_ang'] + df['lattice_vector_2_ang'] + df['lattice_vector_3_ang'])

    df['NTA_A_B_G'] = df['number_of_total_atoms'] * (df['lattice_angle_alpha_degree'] + df['lattice_angle_beta_degree'] + df['lattice_angle_gamma_degree'])

    

new_feat(train)  

new_feat(test)    
print(len(train.columns))

print(len(test.columns))
col = ['formation_energy_ev_natom','bandgap_energy_ev']

X = train.drop(['id'] + col, axis=1)

y = train[col]

x_test = test.drop(['id'], axis=1)
X.head()
y.head()
x_test.head()
print(X.shape)

print(x_test.shape)
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=2017)

X_train.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.pipeline import Pipeline

pipeline = Pipeline([

    #('features',feats),

    #'gbr', ensemble.GradientBoostingRegressor(random_state = 2017),

    ('rfr', RandomForestRegressor(random_state = 2017)),

    #('gbr', GradientBoostingRegressor(random_state = 2017))

    #('knn', KNeighborsRegressor(n_jobs = -1))

    

])
print(X_train.shape)

print(y_train.shape)
pipeline.fit(X_train, y_train)



preds = pipeline.predict(X_val)

#preds1 = np.expm1(preds)#.clip(lower=0.)

#np.mean(preds == y_val)
from sklearn.model_selection import GridSearchCV



hyperparameters = { 

                    #'gbr__learning_rate': [0.1, 0.15, 0.75],

                    'rfr__n_estimators': [225, 235, 250],

                    'rfr__max_depth': [4, 6, 8],

                    'rfr__min_samples_leaf': [2, 4, 6]

                    #'knn__n_neighbors': [4, 7]

                  }

clf = GridSearchCV(pipeline, hyperparameters, cv = 3)

 

# Fit and tune model

clf.fit(X_train, y_train)
clf.best_params_
predictions = clf.predict(x_test)
predictions.shape
out = pd.DataFrame(predictions)



out.columns = ['formation_energy_ev_natom', 'bandgap_energy_ev']

#out.columns = ['formation_energy_ev_natom']

out.head()               
result = pd.concat([test[['id']], np.expm1(out.formation_energy_ev_natom), np.expm1(out.bandgap_energy_ev)], axis=1)



result.columns = ['id', 'formation_energy_ev_natom', 'bandgap_energy_ev']

result.head()



result.to_csv('gradient boosting.csv', index=False)