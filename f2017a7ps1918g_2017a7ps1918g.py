import numpy as np

import pandas as pd

import matplotlib

import sklearn  
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/bits-f464-l1/train.csv')
modify_dataset = dataset

len(modify_dataset)
##Do not execute these commented out cells



#converting one hot encoding to integers

#a=[None]*(len(modify_dataset))

#for i in range(len(modify_dataset)):

  #a[i] = np.array(modify_dataset.iloc[i,96:103].values)

  #a[i]= (np.argmax(a[i]))
#a[0:10]
#dropping the one hot encoded columns

#modify_dataset=modify_dataset.drop(['a0', 'a1', 'a2','a3','a4','a5','a6'], axis=1)
#adding the integer column which represents agent ID

#modify_dataset.insert(96, "Agent_ID", a, True) 
#removing columns which have only one unique value

to_drop1=[]

for column in modify_dataset:

   # Select column contents by column name using [] operator

   unique_value = modify_dataset[column].nunique()

   print('Column Name : ', column)

   print('Number of unique values : ', unique_value)

   if unique_value==1:

     to_drop1.append(column)

     modify_dataset=modify_dataset.drop([column], axis=1)

#### DONOT EXECUTE ######

#Removing the columns which have a correlation of greater than 0.95



# Create correlation matrix

#corr_matrix = modify_dataset.corr().abs()

#print(corr_matrix)

#print(corr_matrix.shape)



# Select upper triangle of correlation matrix

#upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

#to_drop2 = [column for column in upper.columns if any(upper[column] > 0.95)]



#drop them

#modify_dataset = modify_dataset.drop(modify_dataset[to_drop2], axis=1)
x_train = modify_dataset.iloc[:,1:len(modify_dataset.columns)-1].values
x_train.shape
y_train = modify_dataset.iloc[:,len(modify_dataset.columns)-1:len(modify_dataset.columns)].values
### DONOT EXECUTE ####

#Removing NaNs and replacing them with mean values

#from sklearn.impute import SimpleImputer

#imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

#imputer=imputer.fit(x_train)

#x_train=imputer.transform(x_train)
### DONOT EXECUTE ####

#Scaling the data

#from sklearn.preprocessing import StandardScaler

#sc = StandardScaler()

#x_train = sc.fit_transform(x_train)
### DONOT EXECUTE ####

#model 7

#from sklearn.ensemble import RandomForestRegressor

#from sklearn.ensemble import BaggingRegressor 

#model = BaggingRegressor(base_estimator=RandomForestRegressor(n_estimators=100,random_state=42)).fit(x_train, y_train)
### DONOT EXECUTE ####

#model 5

#from sklearn.experimental import enable_hist_gradient_boosting  # noqa

#from sklearn.ensemble import HistGradientBoostingRegressor

#model = HistGradientBoostingRegressor(max_leaf_nodes=None,max_iter=300,l2_regularization=1).fit(x_train,y_train)
### DONOT EXECUTE ####

#Model 4

#from sklearn.linear_model import RidgeCV

#from sklearn.linear_model import Lasso

#from sklearn.ensemble import AdaBoostRegressor

#from sklearn.svm import LinearSVR

#from sklearn.ensemble import RandomForestRegressor

#from sklearn.ensemble import StackingRegressor

#from sklearn.tree import DecisionTreeRegressor

#from sklearn.experimental import enable_hist_gradient_boosting  # noqa

#from sklearn.ensemble import HistGradientBoostingRegressor

#estimators = [('lr', RidgeCV()),('rf',RandomForestRegressor(n_estimators=50,random_state=42)),('svr', LinearSVR(random_state=42),('ar',AdaBoostRegressor(random_state=0, n_estimators=300,learning_rate=1.0,loss='square',base_estimator=DecisionTreeRegressor(max_depth=None,random_state=0))))]

#model = StackingRegressor(estimators=estimators,final_estimator=HistGradientBoostingRegressor(max_leaf_nodes=None,max_iter=200,l2_regularization=1)).fit(x_train,y_train)
### DONOT EXECUTE ####

#Model 3

#from sklearn import ensemble

#params = {'n_estimators': 700, 'max_depth': 4, 'min_samples_split': 10,

          #'learning_rate': 0.01, 'loss': 'ls'}

#regr = ensemble.GradientBoostingRegressor(**params)



#Model 2

from sklearn.ensemble import AdaBoostRegressor

#from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

regr = AdaBoostRegressor(random_state=0, n_estimators=300,learning_rate=1.0,loss='square',base_estimator=DecisionTreeRegressor(max_depth=None,random_state=0))

### DONOT EXECUTE ####

#Model 1

#using a random forest regressor

#from sklearn.ensemble import RandomForestRegressor

#regr = RandomForestRegressor(n_estimators=50,random_state=0) 

#model = regr.fit(x_train, y_train)
model = regr.fit(x_train,y_train)
#### TEST DATA PREPROCESSING AND PREDICTION ####
test_dataset = pd.read_csv('../input/bits-f464-l1/test.csv')
test_modify_dataset = test_dataset

len(test_modify_dataset)
### DONOT EXECUTE ####

#converting one hot encoding to integers

#b=[None]*(len(test_modify_dataset))

#for i in range(len(test_modify_dataset)):

  #b[i] = np.array(test_modify_dataset.iloc[i,96:103].values)

  #b[i]= (np.argmax(b[i]))
### DONOT EXECUTE ####

#dropping the one hot encoded columns

#test_modify_dataset=test_modify_dataset.drop(['a0', 'a1', 'a2','a3','a4','a5','a6'], axis=1)
### DONOT EXECUTE ####

#adding the integer column which represents agent ID

#test_modify_dataset.insert(96, "Agent_ID", b, True) 
test_modify_dataset = test_modify_dataset.drop(test_modify_dataset[to_drop1], axis=1)

x_test = test_modify_dataset.iloc[:,1:len(test_modify_dataset.columns)].values
x_test.shape
#imputer=imputer.fit(x_test)

#x_test=imputer.transform(x_test)
#x_test = sc.fit_transform(x_test)
#Predict the regressed values

predictions = model.predict(x_test) 
predictions.shape
x_test.shape
predictions
test_modify_dataset
submission = pd.DataFrame()

submission['id']=test_dataset.iloc[:,0].values

submission['label']=predictions
submission.to_csv('lab1_submission.csv',index=False)