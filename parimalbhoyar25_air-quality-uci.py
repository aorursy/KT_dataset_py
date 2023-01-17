import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

from pandas import Series, DataFrame



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('/kaggle/input/airquality-uci/AirQualityUCI.csv', sep=',', delimiter=";",decimal=",")
data.head()
data.tail()
data.shape
data.info()
#Deleting the Unnamed: 15 and Unnamed: 16 columns.

data = data.drop(["Unnamed: 15","Unnamed: 16"], axis=1)
data.head()
data.shape
data.info()
data.isnull().any()
data.isnull().sum()
#Deleting all Null values in our dataset permanently.

data.dropna(inplace=True)

data.shape
data.set_index("Date", inplace=True) 

#setting Date column as new index of out dataframe.
data.head(1)
data.index = pd.to_datetime(data.index) #Converting the index in datetime datatype.

type(data.index)
data.head(1)
data['Time'] = pd.to_datetime(data['Time'],format= '%H.%M.%S').dt.hour #Selecting only Hour value from the 'Time' Column.

type(data['Time'][0])
data.head()
data.info()
data.describe()
data.plot.box()

plt.xticks(rotation = 'vertical')

plt.show()
data.replace(to_replace= -200, value= np.NaN, inplace= True)
data.isnull().any()
data.isnull().sum()
plt.figure(figsize=(8,6))

sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
data.drop('NMHC(GT)', axis=1, inplace=True)
data.describe()
data.shape
data.fillna(data.median(), inplace=True)
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
sns.set_style('whitegrid')

eda_data = data.drop(['Time','RH','AH','T'], axis=1)

sns.pairplot(eda_data)
data.hist(figsize = (20,20))

plt.show()
data.drop(['Time','RH','AH','T'], axis=1).resample('M').mean().plot(figsize = (20,8))

plt.legend(loc=1)

plt.xlabel('Month')

plt.ylabel('All Toxic Gases in the Air')

plt.title("All Toxic Gases' Frequency by Month")
data['NOx(GT)'].resample('M').mean().plot(kind='bar', figsize=(18,6))

plt.xlabel('Month')

plt.ylabel('Total Nitrogen Oxides (NOx) in ppb')   # Parts per billion (ppb)

plt.title("Mean Total Nitrogen Oxides (NOx) Level by Month")
plt.figure(figsize=(20,6))

sns.barplot(x='Time',y='NOx(GT)',data=data, ci=False)

plt.xlabel('Hours')

plt.ylabel('Total Nitrogen Oxides (NOx) in ppb') # Parts per billion (ppb)

plt.title("Mean Total Nitrogen Oxides (NOx) Frequency During Days")
data.plot(x='NO2(GT)',y='NOx(GT)', kind='scatter', figsize = (10,6), alpha=0.3)

plt.xlabel('Level of Nitrogen Dioxide')

plt.ylabel('Level of Nitrogen Oxides (NOx) in ppb') # Parts per billion (ppb)

plt.title("Mean Total Nitrogen Oxides (NOx) Frequency During Days")

plt.tight_layout();
plt.figure(figsize=(10,8))

sns.heatmap(data.corr(), annot=True, linewidths=.20)
sns.kdeplot(data['NOx(GT)'])

plt.show()
plt.figure(figsize=(20,5))

plt.plot(data['NOx(GT)'])

plt.show()
plt.figure(figsize=(21,8))

plt.plot(data['NO2(GT)'])

plt.show()
data.shape
X = data.drop(['NOx(GT)','Time'], axis=1)



y= data['NOx(GT)']
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.preprocessing import RobustScaler



sc=RobustScaler()

x_train=sc.fit_transform(x_train)

x_test=sc.transform(x_test)
from sklearn.linear_model import LinearRegression



lm = LinearRegression()

lm.fit(x_train, y_train)
print(lm.intercept_)
coeff_data = pd.DataFrame(lm.coef_, index=X.columns, columns=['Coefficient'])

coeff_data
prediction = lm.predict(x_test)

plt.scatter(y_test, prediction, c="blue", alpha=0.3)

plt.xlabel('Measured')

plt.ylabel('Predicted')

plt.title('Linear Regression Predicted vs Actual')
score_train = lm.score(x_train, y_train)

score_train
prediction = lm.predict(x_test)
score_test = lm.score(x_test, y_test)

score_test
sns.distplot((y_test-prediction), bins=70, color="purple")

from sklearn import metrics

print('MAE:',metrics.mean_absolute_error(y_test, prediction))

print('MSE:',metrics.mean_squared_error(y_test, prediction))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, prediction)))
coeff_data
from sklearn.neighbors import KNeighborsRegressor

knn=KNeighborsRegressor(n_neighbors=5)

knn.fit(x_train,y_train)
prediction = knn.predict(x_test)

plt.scatter(y_test, prediction, c="black", alpha=0.3)

plt.xlabel('Measured')

plt.ylabel('Predicted')

plt.title('K-nearest Neighbors Predicted vs Actual')
knn_train = knn.score(x_train,y_train)

knn_train
kreg_test = knn.score(x_test,y_test)

kreg_test
from sklearn.neighbors import KNeighborsRegressor

kreg=KNeighborsRegressor()
para={'n_neighbors':np.arange(1,51),'weights':['uniform','distance'],

      'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],'leaf_size':np.arange(10,51)}
from sklearn.model_selection import RandomizedSearchCV
knn_cv=RandomizedSearchCV(kreg,para,cv=5,random_state=0)

knn_cv.fit(x_train,y_train)
print(knn_cv.best_score_)

print(knn_cv.best_params_)
kn=KNeighborsRegressor(weights='distance', n_neighbors= 12, leaf_size= 18, algorithm= 'brute')

kn.fit(x_train,y_train)
prediction = kn.predict(x_test)

plt.scatter(y_test, prediction, c="black", alpha=0.3)

plt.xlabel('Measured')

plt.ylabel('Predicted')

plt.title('K-nearest Neighbors(Hyper) Predicted vs Actual')
#Accuracy on Training data

knn_train=kn.score(x_train,y_train)

knn
#Accuracy on Training data

knn_test=knn.score(x_test,y_test)

knn_test
from sklearn.tree import DecisionTreeRegressor

dreg=DecisionTreeRegressor()

dreg.fit(x_train,y_train)
prediction = dreg.predict(x_test)

plt.scatter(y_test, prediction, c="green", alpha=0.3)

plt.xlabel('Measured')

plt.ylabel('Predicted')

plt.title('Decision Tree Predicted vs Actual')
dreg_train = dreg.score(x_train,y_train)

dreg_train
dreg_test = dreg.score(x_test,y_test)

dreg_test
from sklearn.tree import DecisionTreeRegressor

d=DecisionTreeRegressor()
Param_grid={'splitter':['best','random'],'max_depth':[None,2,3,4,5],'min_samples_leaf':np.arange(1,9),

            'criterion':['mse','friedman_mse','mae'],'max_features':['auto','sqrt','log2',None]}
dt_cv=RandomizedSearchCV(d,Param_grid,cv=5,random_state=0)

dt_cv.fit(x_train,y_train)
print(dt_cv.best_score_)

print(dt_cv.best_params_)
dtr=DecisionTreeRegressor(splitter= 'best', min_samples_leaf= 8, max_features= 'auto', max_depth=None, criterion= 'mse')

dtr.fit(x_train,y_train)
prediction = dtr.predict(x_test)

plt.scatter(y_test, prediction, c="green", alpha=0.3)

plt.xlabel('Measured')

plt.ylabel('Predicted')

plt.title('Decision Tree(Hyper) Predicted vs Actual')
dtr_train=dtr.score(x_train,y_train)

dtr_train
#Accuracy on Test Data

dtr_test=dtr.score(x_test,y_test)

dtr_test
from sklearn.ensemble import RandomForestRegressor

rfreg=RandomForestRegressor(n_estimators=10,random_state=0)

rfreg.fit(x_train,y_train)
prediction = rfreg.predict(x_test)

plt.scatter(y_test, prediction, c="purple", alpha=0.3)

plt.xlabel('Measured')

plt.ylabel('Predicted')

plt.title('Random Forest Predicted vs Actual')
rfreg_train = rfreg.score(x_train,y_train)

rfreg_train
rfreg_test = rfreg.score(x_test,y_test)

rfreg_test
rh=RandomForestRegressor()
par={'n_estimators':np.arange(1,91),'criterion':['mse','mae'],'max_depth':[2,3,4,5,None],

     'min_samples_leaf':np.arange(1,9),'max_features':['auto','sqrt','log2',None]}
rh_cv=RandomizedSearchCV(rh,par,cv=5,random_state=0)

rh_cv.fit(x_train,y_train)
print(rh_cv.best_score_)

print(rh_cv.best_params_)
reg=RandomForestRegressor(n_estimators= 74, min_samples_leaf= 2, max_features= 'log2', max_depth=None, criterion= 'mse')

reg.fit(x_train,y_train)
prediction = reg.predict(x_test)

plt.scatter(y_test, prediction, c="purple", alpha=0.3)

plt.xlabel('Measured')

plt.ylabel('Predicted')

plt.title('Random Forest(Hyper) Predicted vs Actual')
#Accuracy on Training data

reg_train=reg.score(x_train,y_train)

reg_train
#Accuracy on Training data

reg_test=reg.score(x_test,y_test)

reg_test
from sklearn.svm import SVR

sreg = SVR(kernel='linear')

sreg.fit(x_train, y_train)
prediction = sreg.predict(x_test)

plt.scatter(y_test, prediction, c="brown", alpha=0.3)

plt.xlabel('Measured')

plt.ylabel('Predicted')

plt.title('SVM Predicted vs Actual')
sreg_train = sreg.score(x_train,y_train)

sreg_train
sreg_test = sreg.score(x_test,y_test)

sreg_test
from xgboost import XGBRegressor

xreg = XGBRegressor()

xreg.fit(x_train, y_train)
plt.scatter(y_test, xreg.predict(x_test),c="red", alpha=0.2)

plt.xlabel('NOx(GT)(y_test)')

plt.ylabel('XGBoost Predicted vs Actual')

plt.show()
xreg_train = xreg.score(x_train,y_train)

xreg_train
xreg_test = xreg.score(x_test,y_test)

xreg_test
booster=['gbtree','gblinear']

base_score=[0.25,0.5,0.75,1]
## Hyper Parameter Optimization



n_estimators = [100, 500, 900, 1100, 1500]

max_depth = [2, 3, 5, 10, 15]

booster=['gbtree','gblinear']

learning_rate=[0.05,0.1,0.15,0.20]

min_child_weight=[1,2,3,4]



# Define the grid of hyperparameters to search

hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth':max_depth,

    'learning_rate':learning_rate,

    'min_child_weight':min_child_weight,

    'booster':booster,

    'base_score':base_score

    }
# Set up the random search with 4-fold cross validation

from sklearn.model_selection import RandomizedSearchCV

random_cv = RandomizedSearchCV(estimator=xreg,

            param_distributions=hyperparameter_grid,

            cv=5, n_iter=50,

            scoring = 'neg_mean_absolute_error',n_jobs = -1,

            verbose = 5, 

            return_train_score = True,

            random_state=42)
random_cv.fit(x_train,y_train)
random_cv.best_estimator_
from xgboost import XGBRegressor

xreg = XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0,

             importance_type='gain', learning_rate=0.1, max_delta_step=0,

             max_depth=10, min_child_weight=3, missing=None, n_estimators=100,

             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

             silent=None, subsample=1, verbosity=1)

xreg.fit(x_train, y_train)
plt.scatter(y_test, xreg.predict(x_test),c="red", alpha=0.2)

plt.xlabel('NOx(GT)(y_test)')

plt.ylabel('HXBoost(Hyper) Predicted vs Actual')

plt.show()
#Accuracy on Training data

xreg_train = xreg.score(x_train,y_train)

xreg_train
#Accuracy on Test Data

xreg_test = xreg.score(x_test,y_test)

xreg_test
results = pd.DataFrame({'Algorithm':['Linear Regression','K-Nearest Neighbour Regressor','Decision Tree Regressor', 'Random Forest Regressor',

                                     'Support Vector Machine Regressor', 'XGBoost'],

                        'Train Accuracy':[score_train, knn_train, dtr_train,reg_train, sreg_train,xreg_train],

                        'Test Accuracy':[score_test, knn_test, dtr_test,reg_test, sreg_test,xreg_test]})

results.sort_values('Test Accuracy', ascending=False)
from sklearn.ensemble import BaggingRegressor
m1=BaggingRegressor(LinearRegression()) #one method

m1.fit(x_train,y_train)
m2=BaggingRegressor(KNeighborsRegressor(weights='distance', n_neighbors= 12, leaf_size= 18, algorithm= 'brute'))

m2.fit(x_train,y_train)

  
m3=BaggingRegressor(DecisionTreeRegressor(splitter= 'best', min_samples_leaf= 8, max_features= 'auto', max_depth=None, criterion= 'mse')) #one method

m3.fit(x_train,y_train)

m4=BaggingRegressor(RandomForestRegressor(n_estimators= 74, min_samples_leaf= 2, max_features= 'log2', max_depth=None, criterion= 'mse')) #one method

m4.fit(x_train,y_train)

#here low bias and low variance than decision tree algo
m5=BaggingRegressor(SVR(kernel='linear')) #one method

m5.fit(x_train,y_train)

m6=BaggingRegressor(XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0,

             importance_type='gain', learning_rate=0.1, max_delta_step=0,

             max_depth=10, min_child_weight=3, missing=None, n_estimators=100,

             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

             silent=None, subsample=1, verbosity=1))

m6.fit(x_train,y_train)
results = pd.DataFrame({'Algorithm':['Linear Regression','K-Nearest Neighbour Regressor','Decision Tree Regressor', 'Random Forest Regressor',

                                     'Support Vector Machine Regressor', 'XGBoost'],

                        'Train Accuracy':[m1.score(x_train,y_train), m2.score(x_train,y_train), m3.score(x_train,y_train),

                                          m4.score(x_train,y_train), m5.score(x_train,y_train), m6.score(x_train,y_train)],

                        'Test Accuracy':[m1.score(x_test,y_test) , m2.score(x_test,y_test), m3.score(x_test,y_test),

                                         m4.score(x_test,y_test), m5.score(x_test,y_test), m6.score(x_test,y_test)]})

results.sort_values('Test Accuracy', ascending=False)