# ignore warnings 
import warnings
warnings.filterwarnings('ignore')

# import general packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
%matplotlib inline
color = sns.color_palette()


# algorithms
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


# modeling helper functions
from sklearn.model_selection import GridSearchCV , KFold , cross_val_score


# to read / write access to some common neuroimaging file formats
## for more information, please visit : https://pypi.org/project/nibabel/
import nibabel

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# print all files in the data folder
base_url = '/kaggle/input/trends-assessment-prediction'
print(os.listdir(base_url))
%time
loading_df = pd.read_csv(base_url +'/loading.csv')
sample_submission = pd.read_csv(base_url +'/sample_submission.csv')
train_df = pd.read_csv(base_url +'/train_scores.csv')
print(f'Size of loading_df : {loading_df.shape}')
print(f'Size of sample_submission : {sample_submission.shape}')
print(f'Size of train_df : {train_df.shape}')
print(f'Size of test set : {len(sample_submission)/5}')
def preview(df):
    print(df.head())
preview(loading_df)
preview(sample_submission)
preview(train_df)
missing_train_df = train_df.isnull().mean() * 100
missing_train_df.sort_values(ascending=False)
loading_df.isnull().sum().sum()
sample_submission.isnull().sum().sum()
target_labels = list(train_df.columns[1:])
target_labels
x = train_df['age']
plt.figure(figsize=(8,6))
plt.hist(x, bins=25, color='g')
plt.xlabel('Age')
plt.ylabel('Number of patients')
plt.title('Age distribution of patients', fontsize = 16)
plt.show()
x = train_df['domain1_var1']
plt.figure(figsize=(8,6))
plt.hist(x, bins=25, color='c')
plt.xlabel('domain1_var1')
plt.ylabel('Number of patients')
plt.title('domain1_var1 distribution', fontsize = 16)
plt.show()
train_df['domain1_var1'].fillna(train_df['domain1_var1'].mean(), inplace=True)
x = train_df['domain1_var2']
plt.figure(figsize=(8,6))
plt.hist(x, bins=25, color='pink')
plt.xlabel('domain1_var2')
plt.ylabel('Number of patients')
plt.title('domain1_var2 distribution', fontsize = 16)
plt.show()
train_df['domain1_var2'].fillna(train_df['domain1_var2'].median(), inplace=True)
x = train_df['domain2_var1']
plt.figure(figsize=(8,6))
plt.hist(x, bins=25, color='y')
plt.xlabel('domain2_var1')
plt.ylabel('Number of patients')
plt.title('domain2_var1 distribution', fontsize = 16)
plt.show()
train_df['domain2_var1'].fillna(train_df['domain2_var1'].mean(), inplace=True)
x = train_df['domain2_var2']
plt.figure(figsize=(8,6))
plt.hist(x, bins=25, color='r')
plt.xlabel('domain2_var2')
plt.ylabel('Number of patients')
plt.title('domain2_var2 distribution', fontsize = 16)
plt.show()
train_df['domain2_var2'].fillna(train_df['domain2_var2'].mean(), inplace=True)
train_df.isnull().sum()
cols = train_df.columns[1:]
correlation = train_df[cols].corr()
mask = np.zeros_like(correlation)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(12,8))
    ax = sns.heatmap(correlation, mask=mask, vmax=.3, square=True, annot=True, cmap='YlGnBu', fmt='.2f', linecolor='white')
    ax.set_title('Correlation Heatmap of training dataset')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=30)           
    plt.show()
sns.pairplot(train_df[cols], kind='scatter', diag_kind='hist', palette='Rainbow')
plt.show()
train_ids = sorted(loading_df[loading_df['Id'].isin(train_df.Id)]['Id'].values)
test_ids = sorted(loading_df[~loading_df['Id'].isin(train_df.Id)]['Id'].values)
predictions = pd.DataFrame(test_ids, columns=['Id'], dtype=str)
features = ('age', 'domain1_var1', 'domain1_var2','domain2_var1','domain2_var2')
data = pd.merge(loading_df, train_df, on='Id')
X_train = data.drop(list(features), axis=1).drop('Id', axis=1)
y_train = data[list(features)]
X_test = loading_df[loading_df.Id.isin(test_ids)].drop('Id', axis=1)

names = ["Linear Regression", "Decision Tree", "Random Forest", "Neural Net" ]    
regressors = [
    LinearRegression(),
    DecisionTreeRegressor(max_depth=5),
    RandomForestRegressor(max_depth=5, n_estimators=10, max_features=1),
    MLPRegressor(alpha=1, max_iter=1000),
    ]
# iterate over classifiers and calculate cross-validation score
for name, reg in zip(names, regressors):
    scores = cross_val_score(reg, X_train, y_train, cv = 5, scoring='neg_mean_absolute_error')
    print(name , ':{:.4f}'.format(scores.mean()))