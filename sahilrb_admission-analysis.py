# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')

df.columns = ['SerialNo', 'GRE_Score', 'TOEFL_Score', 'Uni_Rating', 'SOP', 'LOR', 'CGPA', 'Research', 'Chance_of_Admit']

df.set_index('SerialNo', inplace = True)

df.head()
df.shape
import missingno

missingno.matrix(df)
plt.figure(figsize = (14,7))



sns.heatmap(df.corr(), annot = True)

# Chance of admission shows strong correlation with CGPA, Toefl score and GRE score
plt.figure(figsize = (12,6))

plt.title('Correlation of Each feature with Chance of Admission', fontsize = 14)

sns.barplot(x = df.corr()['Chance_of_Admit'], y = df.columns)
for i, col in enumerate(df.columns):

    print(i, col, '-', df[col].dtype)
print('Counting the number of students in each category')

plt.figure(figsize = (10,6))

sns.countplot(x = 'Research', data = df)
plt.figure(figsize = (12,6))

sns.swarmplot(x = df['Research'], y = df['Chance_of_Admit'])

plt.xlabel('Research', fontsize=16)

plt.ylabel('Chance of Admission', fontsize = 16)

plt.figure(figsize = (10,6))

sns.barplot(x = df['Research'], y = df['Chance_of_Admit'])

plt.xlabel('Research', fontsize=16)

plt.ylabel('Chance of Admission', fontsize = 16)
df[['Research','Uni_Rating', 'Chance_of_Admit']].groupby(['Research'], as_index = True).mean()
plt.figure(figsize = (10,6))

sns.scatterplot(x = df['CGPA'], y = df['Chance_of_Admit'])


sns.lmplot(data = df, x = 'CGPA', y = 'Chance_of_Admit', hue = 'Research', height = 8)

bins = np.linspace(min(df['CGPA']), max(df['CGPA']), 5)

group_names = [1,2,3,4]

df['CGPA_binned'] = pd.cut(df['CGPA'], bins, labels = group_names, include_lowest = True)
plt.figure(figsize = (12,6))

sns.barplot(x = df['CGPA_binned'], y = df['Chance_of_Admit'])

plt.xlabel('CGPA_binned', fontsize = 16)

plt.ylabel('Chance of admission', fontsize = 16)
df['CGPA_binned'] = df['CGPA_binned'].astype(int)

print(df['CGPA_binned'].dtype)
df[['LOR', 'Chance_of_Admit']].groupby('LOR').mean()
plt.figure(figsize = (12,6))

sns.barplot(x = df['LOR'], y = df['Chance_of_Admit'])

plt.xlabel('Letter of Recommendation Strength', fontsize = 16)

plt.ylabel('Chance of admission', fontsize = 16)
df[['SOP', 'Chance_of_Admit']].groupby('SOP').mean()
plt.figure(figsize = (12,6))

sns.barplot(x = df['SOP'], y = df['Chance_of_Admit'])

plt.xlabel('Statement of Purpouse Strength', fontsize = 16)

plt.ylabel('Chance of admission', fontsize = 16)
plt.figure(figsize = (12, 6))

sns.swarmplot(x = df['Uni_Rating'] ,y = df['Chance_of_Admit'], hue = df['CGPA_binned'])

plt.xlabel('University Rating')

plt.ylabel('Chance of Admission')
plt.figure(figsize = (12, 6))

sns.swarmplot(x = df['Uni_Rating'] ,y = df['Chance_of_Admit'], hue = df['Research'])

plt.xlabel('University Rating')

plt.ylabel('Chance of Admission')
df[['Uni_Rating', 'Chance_of_Admit', 'CGPA_binned']].corr()
plt.figure(figsize = (12,6))

plt.title('TOEFL score vs Chances of Admission')

sns.regplot(x = df['TOEFL_Score'], y = df['Chance_of_Admit'])
plt.figure(figsize = (12,6))

sns.scatterplot(x = df['GRE_Score'], y = df['Chance_of_Admit'], hue = df['CGPA_binned'], palette = sns.color_palette('bright', 4))
plt.figure(figsize = (12,6))

sns.regplot(x = df['GRE_Score'], y = df['Chance_of_Admit'])
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import GradientBoostingRegressor

from sklearn import preprocessing

from sklearn.model_selection import RandomizedSearchCV
features = ['GRE_Score', 'TOEFL_Score', 'Uni_Rating', 'SOP', 'LOR', 'Research', 'CGPA']

X = df[features].copy()

X.head()

scalar = preprocessing.StandardScaler()

scalar.fit(X)

X = scalar.transform(X)

X
def check_model_accuracy(model, X, y):

    scores = cross_val_score(model, X, y, cv = 5, scoring = 'r2')

    return scores.mean()
def check_model_error(model, X, y):

    scores = cross_val_score(model, X, y, cv = 5, scoring = 'neg_root_mean_squared_error')

    return scores.mean()
parameters_knn = {

    'n_neighbors':[i for i in range(5,25)],

    'weights':['uniform','distance'],

    'metric':['euclidean', 'manhattan']

}

randm_knn = RandomizedSearchCV(estimator = KNeighborsRegressor(), param_distributions = parameters_knn, cv = 5, n_iter = 15, n_jobs = -1, scoring = 'r2')

randm_knn.fit(X,df.Chance_of_Admit)

print('Best KNN parameters:', randm_knn.best_params_)

print('Best KNN Accuracy:', randm_knn.best_score_)

final_knn_model = KNeighborsRegressor(weights = randm_knn.best_params_['weights'], n_neighbors = randm_knn.best_params_['n_neighbors'], metric = randm_knn.best_params_['metric'] )

print('Error:', -1*check_model_error(final_knn_model, X, df.Chance_of_Admit))
parameters_rf = {

    'n_estimators' : [ i for i in range(100,1001,100)],

    'max_features' : ['auto', 'sqrt', 'log2'],

    'min_samples_leaf' : [1,2,5,10] #The minimum number of samples required to be at a leaf node.

}

randm_rf = RandomizedSearchCV(estimator = RandomForestRegressor(), param_distributions = parameters_rf, cv = 5, n_iter = 15, n_jobs = -1, scoring = 'r2')

randm_rf.fit(X, df.Chance_of_Admit)
print('Best RandomForests Parameters:', randm_rf.best_params_)

print('Best RandomForests Score:', randm_rf.best_score_)

final_randomforests_model = RandomForestRegressor(n_estimators = randm_rf.best_params_['n_estimators'], min_samples_leaf = randm_rf.best_params_['min_samples_leaf'], max_features = randm_rf.best_params_['max_features'])

print('Error:', -1*check_model_error(final_randomforests_model, X, df.Chance_of_Admit))
linearRegression = LinearRegression()

print('Best linear Regression score:', check_model_accuracy(linearRegression, X, df.Chance_of_Admit))

print('Error:', -1*check_model_error(linearRegression, X, df.Chance_of_Admit))
parameters_gb = {

    'learning_rate' : np.arange(0.1,0.91,0.1),

    'n_estimators' : range(100,1001,100),

    'max_depth' : range(2,6),

    'max_features' : ['auto', 'sqrt', 'log2',None]

    

}

randm_gb = RandomizedSearchCV(estimator = GradientBoostingRegressor(), param_distributions = parameters_gb, cv = 5, n_iter = 15, n_jobs = -1, scoring = 'r2')

randm_gb.fit(X, df.Chance_of_Admit)
print('Best GB parameters:', randm_gb.best_params_)

print('Best GB score:', randm_gb.best_score_)

final_GB_model = GradientBoostingRegressor(n_estimators = randm_gb.best_params_['n_estimators'], max_features = randm_gb.best_params_['max_features'], max_depth = randm_gb.best_params_['max_depth'], learning_rate = randm_gb.best_params_['learning_rate'])

print('Error:', -1*check_model_error(final_GB_model, X, df.Chance_of_Admit))
scores = []

models = [final_knn_model, final_randomforests_model, linearRegression, final_GB_model]

for model in models:

    scores.append(check_model_accuracy(model, X, df.Chance_of_Admit))

    
plt.figure(figsize = (14,7))

plt.title('Comparison of Models')

X = ['KNN', 'RandomForests', 'LinearRegression', 'GradientBoosting']

plt.ylabel('Accuracy')

sns.barplot(x=X, y=scores)