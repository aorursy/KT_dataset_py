#Importing the necessary libraries

import numpy as np 
import pandas as pd 
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.ensemble import *
from sklearn.feature_selection import *
from sklearn.metrics import *
from sklearn.tree import *
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.linear_model import *
import xgboost as xgb

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
#To avoid unnecessary warning
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
        


train = pd.read_csv('../input/used-cars-price-prediction/train-data.csv',index_col= 0)
train = train.reindex(np.random.permutation(train.index))
print("TRAIN SHAPE: ",train.shape)
train.info()
train.head()
test = pd.read_csv('../input/used-cars-price-prediction/test-data.csv',index_col= 0)
print(test.shape)
print(test.info())
test.head()
#percentage of missing values
percent_missing = train.isnull().sum() * 100 / len(train)
print(percent_missing)
percent_missing = test.isnull().sum() * 100 / len(test)
print(percent_missing)
#dropping the "New_Price" column that has 86.3% of missing values 

train.drop(columns =['New_Price'],axis =1, inplace = True)
test.drop(columns =['New_Price'],axis =1, inplace = True)
#Mileage attribute has the least percentage of missing values. Let's fill them up manually.

train[train['Mileage'].isnull()]


#Thanks to Google!

train.loc[4904, 'Mileage']  = '23.91 kmpl' 
train.loc[4446, 'Mileage']  = '140 kmpl'
#Now,let's drop the rest of the rows with missing values 


train.dropna(how ='any',inplace = True)
test.dropna(how ='any',inplace = True)
#CHECKING IF ALL THE MISSING VALUES ARE TAKEN CARE OF
train.info()

test.info()
#Mileage - Before we change the datatype, we must extract the actual mileage in numbers without the "Kmpl" 
train['Mileage']= train['Mileage'].str[:-5]
train['Mileage']=train['Mileage'].astype(float);

test['Mileage']= test['Mileage'].str[:-5]
test['Mileage']=test['Mileage'].astype(float);
#Engine - Before we change the datatype, we must extract the actual engine cc in numbers without the "CC" string 

train['Engine'] = train['Engine'].str.strip('CC')
train['Engine']= train['Engine'].astype(float);

test['Engine'] = test['Engine'].str.strip('CC')
test['Engine']= test['Engine'].astype(float);
train['Power'] = train['Power'].fillna(value = "null")
train["Power"]= train["Power"].replace("null", "NaN")
train['Power'] = train['Power'].str.strip('bhp ')
train['Power'] = train['Power'].astype(float)

train.dropna(how ='any',inplace = True)
test['Power'] = test['Power'].fillna(value = "null")
test["Power"]= test["Power"].replace("null", "NaN")
test['Power'] = test['Power'].str.strip('bhp ')
test['Power'] = test['Power'].astype(float)

test.dropna(how ='any',inplace = True)
#Year
train['Year'] = train['Year'].astype(str)

test['Year'] = test['Year'].astype(str)

#CHECKING
train.info()
test.info()
train.to_csv('trainfinal.csv')
test.to_csv('testfinal.csv')
x = pd.read_csv('trainfinal.csv')
print(x.shape)
x.head()
#dropping the unnamed:0 column
x.drop(columns=['Unnamed: 0'],axis=1,inplace = True)
x["breakdown"] = x.Name.str.split(" ")
x["breakdown"].head()
#Lets store the brand name in our new column
brand_list=[]
for i in range(len(x)):
    a = x.breakdown[i][0]
    brand_list.append(a)

x['Brand'] = brand_list
# We don't need these columns now
x.drop(columns=['Name','breakdown'],axis=1,inplace=True)
#Lets analyse the new attribute
x['Brand'].unique()
duplic = {'ISUZU': 'Isuzu'}
x.replace({"Brand": duplic},inplace = True) 
#CHECKING
x['Brand'].value_counts()

#Sorted!
#Lets encode our categorical values 

labelencoder = LabelEncoder()
label_array=[]

label_array = ['Location','Year','Fuel_Type','Transmission','Owner_Type','Brand']

for ele in label_array:
    x[ele] = labelencoder.fit_transform(x[ele])

#CHECKING
x.head()
#feature selection
X_fs = x[['Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 'Owner_Type',
       'Mileage', 'Engine', 'Power', 'Seats', 'Brand']]

y_fs = x['Price']


y_fs = y_fs*100
y_fs = y_fs.astype(int)


bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(X_fs,y_fs)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X_fs.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
fea = pd.DataFrame(featureScores.nlargest(10,'Score'))
print(featureScores.nlargest(10,'Score'))  #print 10 best features
selection= ExtraTreesRegressor()
selection.fit(X_fs,y_fs)

plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X_fs.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

#Preparing Training set 
X = np.array(x.drop(['Price'],axis = 1)) 
Y = x.Price.values
#splitting into train and test set
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=25)
#selecting best models

model_selc = [LinearRegression(),
             DecisionTreeRegressor(),
             RandomForestRegressor(n_estimators=10),
             KNeighborsRegressor(),
             GradientBoostingRegressor()]

kfold = RepeatedKFold(n_splits=5, n_repeats=10, random_state= None)
cv_results = []
cv_results_mean =[]
for ele in model_selc:
    cross_results = cross_val_score(ele, X_train, Y_train, cv=kfold, scoring ='r2')
   
    cv_results.append(cross_results)
   
    cv_results_mean.append(cross_results.mean())
    print("\n MODEL: ",ele,"\nMEAN R2:",cross_results.mean() )

#Let's try xgboost now
my_xgb = xgb.XGBRegressor(objective='reg:linear',learning_rate = 0.1, n_estimators = 100,verbosity = 0,silent=True)
xgb_results = cross_val_score(my_xgb, X_train, Y_train, cv=kfold, scoring ='r2')
print("\n MODEL: XGBOOST","\nMEAN R2:",xgb_results.mean() )
#We use GridSearch for fine tuning Hyper Parameters

from sklearn.model_selection import *


n_estimator_val = np.arange(100,400,100).astype(int)
max_depth_val = [2,3,4]


grid_params = { 'loss' : ['ls'] ,
               'learning_rate' : [0.1],
               'n_jobs': [-1],
               'n_estimators' : n_estimator_val,
               'max_depth' : max_depth_val
              }
gs = GridSearchCV(xgb.XGBRegressor(silent= True),grid_params,verbose=1,cv=5,n_jobs =-1)
gs_results = gs.fit(X_train,Y_train)
#To Display the Best Score
gs_results.best_score_
#To Display the Best Estimator
gs_results.best_estimator_
#To Display the Best Parameters
gs_results.best_params_
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.8, gamma=0.5,
       importance_type='gain', learning_rate=0.1, max_delta_step=0,
       max_depth=4, min_child_weight=10, missing=None, n_estimators=1900,
       n_jobs=-1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1.0, verbosity=1)
folds = 3
param_comb = 10

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

params = { 'n_jobs': [-1],
        'n_estimators' : n_estimator_val,
        'learning_rate' : [0.1],
        'min_child_weight': [9],
        'gamma': [0.5],
        'subsample': [0.6],
        'colsample_bytree': [0.8, 1.0],
        'max_depth': [3, 4]
        }
xgb_regrsv = xgb.XGBRegressor()

random_search = RandomizedSearchCV(xgb_regrsv, params, n_iter=param_comb, scoring='r2', 
                                   n_jobs=-1, cv=5 )

random_search.fit(X_train, Y_train);
random_search.best_score_
random_search.best_estimator_
xgb_tuned = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.8, gamma=0.5,
       importance_type='gain', learning_rate=0.1, max_delta_step=0,
       max_depth=4, min_child_weight=9, missing=None, n_estimators=300,
       n_jobs=-1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1.0, verbosity=0)
xgb_tuned.fit(X_train,Y_train)
y_pred =xgb_tuned.predict(X_test)

print("Training set accuracy: ",xgb_tuned.score(X_train,Y_train))
print("Test set accuracy    : ",xgb_tuned.score(X_test,Y_test))

print("\t\tError Table")
print('Mean Absolute Error      : ', mean_absolute_error(Y_test, y_pred))
print('Mean Squared  Error      : ', mean_squared_error(Y_test, y_pred))
print('Root Mean Squared  Error : ', np.sqrt(mean_squared_error(Y_test, y_pred)))
print('R Squared Error          : ', r2_score(Y_test, y_pred))
