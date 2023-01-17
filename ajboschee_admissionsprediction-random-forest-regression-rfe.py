import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('../input/Admission_Predict.csv',sep = ",")

print(df.head())
df.describe()
df.info()
sns.distplot(df['GRE Score'])
sns.distplot(df['TOEFL Score'])
sns.distplot(df['CGPA'])
sns.jointplot(x='TOEFL Score',y='GRE Score',data=df,kind='scatter')
sns.jointplot(x='CGPA',y='GRE Score',data=df,kind='scatter')
sns.jointplot(x='Chance of Admit ',y='CGPA',data=df,kind='scatter')
sns.countplot(df['Research'])
sns.lmplot(x='Chance of Admit ',y='LOR ',data=df,hue='Research')
sns.lmplot(x='CGPA',y='LOR ',data=df,hue='Research')
sns.jointplot(x='University Rating',y='CGPA',data=df,kind='scatter')
from sklearn.model_selection import train_test_split



df=df.drop(['Serial No.'], axis=1)



X = df.drop('Chance of Admit ', axis=1)

y = df['Chance of Admit ']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.tree import DecisionTreeRegressor



dtr = DecisionTreeRegressor()

dtr.fit(X_train, y_train)
predict = dtr.predict(X_test)
from sklearn import metrics

from sklearn.metrics import mean_squared_error



dtr_mse= mean_squared_error(y_test, predict)

dtr_rmse = np.sqrt(metrics.mean_squared_error(y_test, predict))



print('Decision Tree Regression RMSE: ', dtr_rmse)

print('Decision Tree Regression MSE: ', dtr_mse)
plt.scatter(predict,y_test)
from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestRegressor



  

rfe = RFE(RandomForestRegressor(n_estimators=50, random_state=42), n_features_to_select=3)



rfe.fit(X_train, y_train)



#create dataframe of feature and ranking. Top 3 have '1' in rfe.ranking_ array

rfe_features_rank = pd.DataFrame({'feature':X_train.columns, 'score':rfe.ranking_})

#compose list of highest ranked features

top_three_features = rfe_features_rank[rfe_features_rank['score'] == 1]['feature'].values

print('Top three features: ', top_three_features)
top3_df = df[[top_three_features[0],top_three_features[1], top_three_features[2], 'Chance of Admit ']]
X = top3_df.drop('Chance of Admit ', axis=1)

y = top3_df['Chance of Admit ']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

dtr.fit(X_train, y_train)

predict = dtr.predict(X_test)
dtr_mse= mean_squared_error(y_test, predict)

dtr_rmse = np.sqrt(metrics.mean_squared_error(y_test, predict))



print('Decision Tree Regression with only top three features RMSE: ', dtr_rmse)

print('Decision Tree Regression with only top three features MSE: ', dtr_mse)
plt.scatter(predict,y_test)