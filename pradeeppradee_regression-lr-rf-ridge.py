import pandas as pd

import numpy as np
df=pd.read_csv('../input/california-housing-prices/housing.csv')
df.head()
df.shape
df.isnull().sum()
df.apply(lambda x:len(x.unique()))
df.info()
import seaborn as sns

sns.catplot('ocean_proximity',kind = 'count',data = df,aspect =3)
sns.distplot(df['longitude'],color='g')
sns.distplot(df['latitude'],color='g')
sns.distplot(df['housing_median_age'],color='g')
sns.distplot(df['total_rooms'],color='g')
sns.distplot(df['population'],color='g')
sns.distplot(df['households'],color='g')
sns.distplot(df['median_income'],color='g')
sns.distplot(df['median_house_value'],color='g')
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df['ocean_proximity']=le.fit_transform(df['ocean_proximity'])
df.head()
#correlation_matrix

import matplotlib.pyplot as plt

corr_m = df.corr() 

f, ax = plt.subplots(figsize =(7,6)) 

sns.heatmap(corr_m,annot=True, cmap ="YlGnBu", linewidths = 0.1) 
#missing values

df['total_bedrooms'] = df['total_bedrooms'].fillna((df['total_bedrooms'].median()))
df.isnull().sum()
df.head()
#model_check_1

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error

from math import sqrt

from sklearn.linear_model import LinearRegression
X=df.drop(['median_house_value'],axis=1)

y=df['median_house_value']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)

L_R= LinearRegression()

L_R.fit(X_train,y_train)

y_pred=L_R.predict(X_test)

r2scores= r2_score(y_test,y_pred)

print("r2scores : ",r2scores)

L_R.score(X_train,y_train),L_R.score(X_test,y_test)

print(L_R.intercept_)
df.head()
df['rooms_per_household'] = df['total_rooms'] / df['households']
x=df.columns
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

for i in x:

    df[i] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(df[i])))
df.head()
#model_1

X=df.drop(['median_house_value','total_rooms'],axis=1)

y=df['median_house_value']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)

L_R= LinearRegression()

L_R.fit(X_train,y_train)

y_pred=L_R.predict(X_test)

r2scores= r2_score(y_test,y_pred)

print("r2scores : ",r2scores)
#rmse(L_R)

mse1=mean_squared_error(y_test,y_pred)

L_R_score=np.sqrt(mse1)

L_R_score
#model_2

from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor()

rf.fit(X_train,y_train)

y_pred2=rf.predict(X_test)

mse2=mean_squared_error(y_test,y_pred2)

rf_score=np.sqrt(mse2)

rf_score
r2scores_2= r2_score(y_test,y_pred2)

r2scores_2
#model_3 (additional)

from sklearn.linear_model import Ridge

rr = Ridge(alpha=0.01)

rr.fit(X_train, y_train) 

y_pred4= rr.predict(X_test)

mse_3=mean_squared_error(y_test,y_pred4)

rid_score=np.sqrt(mse_3)

print(rid_score)

r2scores_3=r2_score(y_test, y_pred4)

print(r2scores_3)
F_scores = {'Model':  ['Linear_Regression', 'RandomForest_Regressor','Ridge'],

            'MSE': [mse1,mse2,mse_3],

            'RMSE': [L_R_score, rf_score,rid_score ],

            'R2': [r2scores,r2scores_2,r2scores_3]}
df_scores = pd.DataFrame (F_scores, columns = ['Model','MSE','RMSE','R2'])

df_scores
print(df_scores.to_markdown(tablefmt="grid"))
#Additional graphs to compare actual vs predicted

df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

ax1 = sns.distplot(df1['Actual'], hist=False, color="red", label="Actual Value")

sns.distplot(df1['Predicted'], hist=False, color="blue", label="Predicted Values" , ax=ax1)
df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred2})

ax1 = sns.distplot(df2['Actual'], hist=False, color="red", label="Actual Value")

sns.distplot(df2['Predicted'], hist=False, color="blue", label="Predicted Values" , ax=ax1)
df3 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred4})

ax1 = sns.distplot(df3['Actual'], hist=False, color="red", label="Actual Value")

sns.distplot(df3['Predicted'], hist=False, color="blue", label="Predicted Values" , ax=ax1)