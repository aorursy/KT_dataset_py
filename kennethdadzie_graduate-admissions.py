# Ignore harmless warnings

import warnings

warnings.filterwarnings("ignore")



#Importing libraries for data analysis and cleaning

import numpy as np

import pandas as pd



#importing visualisation libraries for data visualisation

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.express as px



#load datasets

df = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')

df = df.rename(columns = {'Chance of Admit ' : 'Chance of Admit'})

df.head()
df.info()
#Dropping the serial column (Irrelavant for predictions)

df = df.drop('Serial No.',axis=1)
df.describe()
#Checking for null values

df.isnull().sum()
plt.figure(figsize=(12,5))

sns.distplot(df['Chance of Admit'],bins=30)

plt.show()



plt.figure(figsize=(12,5))

sns.distplot(df['CGPA'],bins=30)

plt.show()



plt.figure(figsize=(12,5))

sns.distplot(df['GRE Score'],bins=30)

plt.show()



plt.figure(figsize=(12,5))

sns.distplot(df['TOEFL Score'],bins=30)

plt.show()
px.pie(df,'Research',title='Research Experience Distribution')
plt.figure(figsize=(12,6))

sns.heatmap(df.corr(),cmap='coolwarm',vmax=1,vmin=-1,annot=True);

# fix for mpl bug that cuts off top/bottom of seaborn viz

b, t = plt.ylim() # discover the values for bottom and top

b += 0.5 # Add 0.5 to the bottom

t -= 0.5 # Subtract 0.5 from the top

plt.ylim(b, t) # update the ylim(bottom, top) values

plt.show() 
#Linear fit for CGPA and Chance of Admission

sns.lmplot(data=df,y='Chance of Admit',x='CGPA');
#Bar chart showing highest correlation with chance of admit

df.corr()['Chance of Admit'].drop('Chance of Admit').sort_values(ascending=False).plot(kind='bar',figsize=(10,5));
#University ranking

px.pie(df,names='University Rating')
df.corr()['University Rating'].drop(['University Rating','Chance of Admit']).sort_values(ascending=False)
plt.figure(figsize=(8,4))

plt.title('PER STATEMENT OF PURPOSE')

sns.barplot(data=df,x='University Rating',y='SOP')

plt.show()



plt.figure(figsize=(8,4))

plt.title('PER CGPA')

sns.barplot(data=df,x='University Rating',y='CGPA')

plt.show()



plt.figure(figsize=(8,4))

plt.title('PER TOEFL Score')

sns.barplot(data=df,x='University Rating',y='TOEFL Score')

plt.show()

plt.figure(figsize=(10,5))

sns.barplot(data=df,x='University Rating',y='Research');
df.head()
#Defining the variables X and y Where; 



#X are the features for training 

X = df.drop('Chance of Admit',axis=1)



#y is the target(Chance of admittance) to be predicted

y = df['Chance of Admit']
#Splitting the train data into train and test purposes.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#Scaling features

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
#mean average of admitance

df['Chance of Admit'].mean()
from sklearn.model_selection import cross_val_score,GridSearchCV,RandomizedSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

import xgboost
#Validation function

def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(estimator=model, X = X_train,y= y_train, scoring="neg_mean_squared_error", cv = 10))

    return(rmse)
#Linear regression deafult params

rmse_cv(LinearRegression()).mean()
#Random forest regressor default params

rmse_cv(RandomForestRegressor()).mean()
#XGBOOST regrssor Default parameters

rmse_cv(xgboost.XGBRegressor()).mean()
LR = LinearRegression()

LR.fit(X_train,y_train)

y_pred = LR.predict(X_test)
from sklearn.metrics import mean_squared_error,explained_variance_score

rmse_lr = np.sqrt(mean_squared_error(y_pred, y_test))

print('RMSE:',np.sqrt(mean_squared_error(y_pred, y_test)))

print('R2:',explained_variance_score(y_pred,y_test))
coeff = pd.DataFrame(LR.coef_,X.columns)

coeff.columns = ['Coefficient']

coeff
sns.distplot((y_test-y_pred),bins=30);
# Our predictions

plt.scatter(y_test,y_pred)



# Perfect predictions

plt.plot(y_test,y_test,'r');
param_grid = {

    'bootstrap': [True,False],

    'max_depth': range(20,200,20),

    'max_features': ['auto','sqrt'],

    'min_samples_leaf': range(1,10,2),

    'min_samples_split': range(1,10,2),

    'n_estimators': range(100,1000,100),

}



# Create a based model

rf = RandomForestRegressor()



# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)
#grid_search.fit(X_train,y_train)

#grid_search.best_estimator
rr = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',

                      max_depth=60, max_features='sqrt', max_leaf_nodes=None,

                      max_samples=None, min_impurity_decrease=0.0,

                      min_impurity_split=None, min_samples_leaf=1,

                      min_samples_split=9, min_weight_fraction_leaf=0.0,

                      n_estimators=100, n_jobs=None, oob_score=False,

                      random_state=None, verbose=0, warm_start=False)
rr.fit(X_train,y_train)
pred_rr = rr.predict(X_test)

rmse_rr = np.sqrt(mean_squared_error(pred_rr, y_test))

print('RMSE:',np.sqrt(mean_squared_error(pred_rr, y_test)))

print('R2:',explained_variance_score(pred_rr,y_test))
# Our predictions

plt.scatter(y_test,pred_rr)



# Perfect predictions

plt.plot(y_test,y_test,'r');
param_tuning = {

        'learning_rate': [0.01, 0.1,0.03],

        'max_depth': [3, 5, 7, 10],

        'min_child_weight': [1, 3, 4,5],

        'subsample': [0.5, 0.7],

        'colsample_bytree': [0.5, 0.6, 0.7],

        'n_estimators' : [100, 200, 500]

}



xgb_model = xgboost.XGBRegressor()



rsearch = RandomizedSearchCV(estimator=xgb_model,

                             param_distributions=param_tuning,

                            n_jobs=-1,cv=5,verbose=1)

 #scoring = 'neg_mean_squared_error',  #MSE



rsearch.fit(X_train,y_train)
rsearch.best_params_
xgr_pred = rsearch.predict(X_test)

rmse_xgb = np.sqrt(mean_squared_error(xgr_pred, y_test))

print('RMSE:',np.sqrt(mean_squared_error(xgr_pred, y_test)))

print('R2:',explained_variance_score(xgr_pred,y_test))
plt.bar(['Linear Reg', 'Random Forest','XGB Reg'], [rmse_lr, rmse_rr,rmse_xgb])

plt.ylabel('RMSE')

plt.title('Compare Models')

plt.show()
print('RMSE:',np.sqrt(mean_squared_error(pred_rr, y_test)))