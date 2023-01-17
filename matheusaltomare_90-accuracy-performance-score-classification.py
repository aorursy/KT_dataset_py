import numpy as np



import pandas as pd

pd.options.plotting.backend = "plotly"



import matplotlib.pyplot as plt



from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import (train_test_split, ShuffleSplit, cross_val_score)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import plot_confusion_matrix



from skopt.space import (Real,

                         Integer,

                         Categorical)

from skopt.utils import use_named_args

from skopt import (gp_minimize, 

                   dump, 

                   load)

from skopt.plots import plot_convergence

from skopt.callbacks import CheckpointSaver



import json
df = pd.read_csv('../input/human-resources-data-set/HRDataset_v13.csv')



# Null rows after 310

df = df.iloc[0:310, :]
df
df.isnull().sum()/df.shape[0]
df = df.loc[:, (df.isnull().sum()/df.shape[0] < 0.3)]
df.isnull().sum()
df[df.isnull().any(axis=1)]
df.dropna(axis=0, inplace=True)

df.reset_index(drop=True, inplace=True)
df.dtypes
df[['PayRate', 'EngagementSurvey', 'EmpSatisfaction', 'SpecialProjectsCount']]
df['PayRate'].plot.hist()
df['EngagementSurvey'].plot.hist()
df['EmpSatisfaction'].plot.hist()
df['SpecialProjectsCount'].plot.hist()
df[['Position', 'State', 'RecruitmentSource', 'ManagerName', 'Sex', 

    'MaritalDesc', 'CitizenDesc', 'HispanicLatino','RaceDesc', 'Department']]
df['Position'].plot.hist()
df['State'].plot.hist()
for state in df['State'].unique():

    if state != 'MA':

        df.replace(state, 'non-MA', inplace=True)
df['State'].plot.hist()
df['RecruitmentSource'].plot.hist()
df['RecruitmentSource'].replace('Pay Per Click', 'Other', inplace=True)

df['RecruitmentSource'].replace('On-line Web application', 'Other', inplace=True)

df['RecruitmentSource'].replace('Careerbuilder', 'Other', inplace=True)

df['RecruitmentSource'].replace('Company Intranet - Partner', 'Other', inplace=True)
df['RecruitmentSource'].plot.hist()
df['ManagerName'].plot.hist()
df['Sex'].plot.hist()
df['MaritalDesc'].plot.hist()
df['CitizenDesc'].plot.hist()
for state in df['CitizenDesc'].unique():

    if state != 'US Citizen':

        df.replace(state, 'non-US Citizen', inplace=True)
df['CitizenDesc'].plot.hist()
df['HispanicLatino'].plot.hist()
df['HispanicLatino'].replace('yes', 'Yes', inplace=True)

df['HispanicLatino'].replace('no', 'No', inplace=True)
df['HispanicLatino'].plot.hist()
df['RaceDesc'].plot.hist()
df['RaceDesc'].replace('American Indian or Alaska Native', 'Others', inplace=True)

df['RaceDesc'].replace('Two or more races', 'Others', inplace=True)

df['RaceDesc'].replace('Hispanic', 'Others', inplace=True)
df['RaceDesc'].plot.hist()
df.drop('HispanicLatino', axis=1, inplace=True)
df['Department'].plot.hist()
print(df.shape)

df.drop(df[df['Department'] == 'Executive Office'].index, axis=0, inplace=True)

print(df.shape)
df['Department'].plot.hist()
df['PerformanceScore'].plot.hist()
df['PerformanceScore'].replace('PIP', 'Bad', inplace=True)

df['PerformanceScore'].replace('Needs Improvement', 'Bad', inplace=True)

df['PerformanceScore'].replace('Fully Meets', 'Good', inplace=True)

df['PerformanceScore'].replace('Exceeds', 'Good', inplace=True)
df['PerformanceScore'].plot.hist()
X = df[['PayRate', 'EngagementSurvey', 'EmpSatisfaction', 'SpecialProjectsCount', 

        'Position', 'State', 'RecruitmentSource', 'ManagerName', 'Sex', 'MaritalDesc', 

        'CitizenDesc', 'RaceDesc', 'Department']]



y = df['PerformanceScore']
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), slice(4, X.shape[1]))],

                       remainder='passthrough')

X = ct.fit_transform(X)
model_name = 'random_forest'
input_shape = X.shape[1]

print(f'input_shape = {input_shape}')



try:

    output_shape = Y.shape[1]

except:

    output_shape = 1

print(f'output_shape = {output_shape}')

      

n_samples = X.shape[0]

print(f'n_samples = {n_samples}')
cv = ShuffleSplit(n_splits=20, test_size=0.20)

cv_opt = ShuffleSplit(n_splits=20, test_size=0.20)
hyperparams_names = ['n_estimators',

                     'max_depth', 

                     'max_features', 

                     'min_samples_split', 

                     'min_samples_leaf']
space  = [Integer(64, 1024, name=hyperparams_names[0]),

          Integer(2, 256, name=hyperparams_names[1]),

          Integer(2, input_shape, name=hyperparams_names[2]),

          Integer(2, 16, name=hyperparams_names[3]),

          Integer(1, 16, name=hyperparams_names[4])]
@use_named_args(space)

def objective(**hyperparams):

    

    print(hyperparams)

        

    cv_scores = cross_val_score(RandomForestClassifier(**hyperparams), 

                                X, y, cv=cv_opt)

    

    return -np.mean(cv_scores)
res_gp = gp_minimize(objective, space, n_calls=100, n_random_starts=20, 

                    random_state=0, verbose=1)
plot_convergence(res_gp)

plt.show()
best_hyperparams = {param:value for param, value in zip(hyperparams_names, res_gp.x)}

print(f'best_hyperparams = {best_hyperparams}')
hyperparams = best_hyperparams
clf = RandomForestClassifier(**hyperparams)

cross_val_scores = pd.DataFrame(cross_val_score(clf, X, y, cv=cv), columns=['Random Forest'])



print(f'cross validation score (accuracy): {round(cross_val_scores.mean().values[0], 2)} +/- {round(cross_val_scores.std().values[0], 2)}')
cross_val_scores.plot.box()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



clf = RandomForestClassifier(**hyperparams)



clf.fit(X_train, y_train)



score = clf.score(X_test, y_test)

print(round(score, 2))
disp = plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues)

disp.ax_.set_title(f'Confusion Matrix  -  accuracy: {round(score, 2)}')