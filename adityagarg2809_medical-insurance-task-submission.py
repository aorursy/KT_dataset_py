



import numpy as np 

import pandas as pd 





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('/kaggle/input/insurance/insurance.csv')
df.head()
df.describe().transpose()
import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline
sns.distplot(df['age'],bins=30)
sns.jointplot(df['age'],df['charges'])
plt.figure(figsize=(14,8))

sns.scatterplot(x=df['age'],y=df['charges'],hue=df['region'])
plt.figure(figsize=(14,8))

sns.scatterplot(x=df['age'],y=df['charges'],hue=df['smoker'])
sns.countplot(df['children'])
plt.figure(figsize=(14,8))

sns.scatterplot(x=df['age'],y=df['charges'],hue=df['children'])
plt.figure(figsize=(14,8))

sns.kdeplot(df['bmi'],shade=True)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()
from sklearn.model_selection import train_test_split
X = df[df.columns[:len(df.columns)-1]]

y = df[df.columns[-1]]
def one_hot_encoder_1(x):

    if x == 'yes':

        return 1

    else:

        return 0
def one_hot_encoder_2(x):

    if x == 'male':

        return 1

    else:

        return 0
df['region'].unique()
X['smoker'] = X['smoker'].apply(one_hot_encoder_1)
X['sex'] = X['sex'].apply(one_hot_encoder_2)
X.head()
X=pd.concat([X,pd.get_dummies(X['region'],drop_first=True)],axis=1)
X = X.drop('region',axis=1)
X.head()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=101)
scaled_train = scaler.fit_transform(X_train)

scaled_test = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression



lr_model = LinearRegression()
lr_model.fit(scaled_train,y_train)
preds = lr_model.predict(scaled_test)
lr_model.coef_
lr_model.intercept_
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print(mean_squared_error(y_test,preds)**0.5)
print(mean_absolute_error(y_test,preds))
print(r2_score(y_test,preds))
from sklearn.preprocessing import PolynomialFeatures
poly_transformer = PolynomialFeatures(3)
X_poly_train,X_poly_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=101)
X_poly_train = poly_transformer.fit_transform(X_poly_train)

X_poly_test = poly_transformer.transform(X_poly_test)
poly_model = LinearRegression()
poly_model.fit(X_poly_train,y_train)
pred = poly_model.predict(X_poly_test)
poly_model.coef_
poly_model.intercept_
print(mean_squared_error(y_test,preds)**0.5)
print(mean_absolute_error(y_test,preds))
print(r2_score(y_test,preds))
from sklearn.ensemble import RandomForestRegressor
losses = []

i_vals = []

for i in range(1,70):

    decision_forest = RandomForestRegressor(n_estimators = i)

    

    decision_forest.fit(scaled_train,y_train)

    

    pred = decision_forest.predict(scaled_test)

    

    i_vals.append(i)

    

    losses.append(mean_squared_error(y_test,pred)**0.5)

    

    
plt.figure(figsize=(10,8))

plt.plot(i_vals,losses)
decision_forest = RandomForestRegressor(n_estimators = 50)



decision_forest.fit(scaled_train,y_train)



pred = decision_forest.predict(scaled_test)
print(mean_squared_error(y_test,pred)**0.5)
print(mean_absolute_error(y_test,pred))
decision_forest.feature_importances_