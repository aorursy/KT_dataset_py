# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import LinearSVR

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from sklearn.metrics import r2_score,accuracy_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/diamonds/diamonds.csv')
df.head()
df.info()
sns.boxplot(data=df)
df.isnull().sum()
df.drop(columns = 'Unnamed: 0', axis=1, inplace=True)
df.head()
df.head()
enc = OrdinalEncoder()

df['color'] = enc.fit_transform(df[['color']])

df.head()
df['cut'].unique()
cut_map = {'Ideal':0,'Premium':1,'Very Good':2,'Good':3,'Fair':4}

df['cut'] = df['cut'].map(cut_map)
df.head()
clarity_map = {'IF':0,'VVS1':1,'VVS2':2,'VS1':3,'VS2':4, 'SI1':5,'SI2':6,'I1':7}

df['clarity'] = df['clarity'].map(clarity_map)
df.head()
scaling_df = preprocessing.MinMaxScaler()

df[['depth','table']] = scaling_df.fit_transform(df[['depth','table']])
df.head()
df_features = df.drop(columns = 'price')

df_target = df['price']
X_train,X_test,y_train,y_test = train_test_split(df_features,df_target, test_size = 0.2, random_state = 2)
models = [] # Creating a list to store the models

accuracy_score = [] # Creating a list to store the model accuracy
def model_performance(model,model_name,features = X_train,target = y_train, test_features = X_test, true_values = y_test):

    models.append(model_name)

    model.fit(features,target)

    y_pred = model.predict(test_features)

    accuracy = r2_score(true_values,y_pred)

    accuracy_score.append(accuracy)

    print(accuracy)

    

    
reg = LinearRegression()

model_performance(reg, 'Linear Regression')
reg = DecisionTreeRegressor()

model_performance(reg, 'Decision Tree Regression')
reg = KNeighborsRegressor()

model_performance(reg, 'KNN Regression')
reg = GaussianNB()

model_performance(reg, 'Naive Bayes Regression')
reg = Ridge()

model_performance(reg, 'Ridge Regression')
reg = Lasso()

model_performance(reg, 'Lasso Regression')
reg = ElasticNet()

model_performance(reg, 'ElasticNet Regression')
reg = LinearSVR()

model_performance(reg, 'LinearSVR Regression')
reg = RandomForestRegressor(n_estimators = 10, random_state = 42)

model_performance(reg, 'Random Forest Regression')
reg = AdaBoostRegressor(n_estimators = 100)

model_performance(reg, 'AdaBoost Regression')
reg = GradientBoostingRegressor(n_estimators = 100, random_state = 42, max_depth=4)

model_performance(reg, 'GradientBoosting Regression')
model_comparision_df = pd.DataFrame({'Regression Model': models, 'R2 Accuracy Score': accuracy_score})

model_comparision_df.sort_values(by = 'R2 Accuracy Score', ascending= False)
sns.barplot(x = 'R2 Accuracy Score', y='Regression Model', data = model_comparision_df)