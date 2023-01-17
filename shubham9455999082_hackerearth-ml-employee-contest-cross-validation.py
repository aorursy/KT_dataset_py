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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/Train.csv')
test=pd.read_csv('/kaggle/input/Test.csv')
test['Age'].fillna(test['Age'].median(), inplace=True)
test['Time_of_service'].fillna(test['Time_of_service'].median(), inplace=True)
test['Work_Life_balance'].fillna(test['Work_Life_balance'].median(), inplace=True)
test['VAR2'].fillna(test['VAR2'].median(), inplace=True)
test['VAR4'].fillna(test['VAR4'].median(), inplace=True)
test['Pay_Scale'].fillna(test['Pay_Scale'].median(), inplace=True)
train['Decision_skill_possess'].fillna('Conceptual', inplace=True)
train['Age'].fillna(train['Age'].median(), inplace=True)
train['Time_of_service'].fillna(train['Time_of_service'].median(), inplace=True)
train['Work_Life_balance'].fillna(train['Work_Life_balance'].median(), inplace=True)
train['VAR2'].fillna(train['VAR2'].median(), inplace=True)
train['VAR4'].fillna(train['VAR4'].median(), inplace=True)
train['Pay_Scale'].fillna(train['Pay_Scale'].median(), inplace=True)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train['Gender']=le.fit_transform(train['Gender'])
test['Gender']=le.transform(test['Gender'])
le1=LabelEncoder()
train['Relationship_Status']=le1.fit_transform(train['Relationship_Status'])
test['Relationship_Status']=le1.transform(test['Relationship_Status'])
le4=LabelEncoder()
train['Hometown']=le4.fit_transform(train['Hometown'])
test['Hometown']=le4.transform(test['Hometown'])
le5=LabelEncoder()
train['Pay_Scale']=le5.fit_transform(train['Pay_Scale'])
test['Pay_Scale']=le5.transform(test['Pay_Scale'])
le2=LabelEncoder()
train['Unit']=le2.fit_transform(train['Unit'])
test['Unit']=le2.transform(test['Unit'])

le3=LabelEncoder()
train['Decision_skill_possess']=le3.fit_transform(train['Decision_skill_possess'])
test['Decision_skill_possess']=le3.transform(test['Decision_skill_possess'])
le6=LabelEncoder()
train['Compensation_and_Benefits']=le6.fit_transform(train['Compensation_and_Benefits'])
test['Compensation_and_Benefits']=le6.transform(test['Compensation_and_Benefits'])
#le11=LabelEncoder()
#train['Attrition_rate']=le11.fit_transform(train['Attrition_rate'])
X=train.iloc[0:,1:23].values
y=train['Attrition_rate'].values
testx=test.iloc[0:, 1:].values
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]
#n_jobs = [-1, 1]
#random_state=  [int(x) for x in np.linspace(start = 0, stop = 10000, num = 10000)]
#criterion=['gini','entropy']
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 20, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 12, 14, 15,16, 17,18]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,6, 8,9]
max_leaf_nodes=[5, 8,9,10, 12,15 , None]
#random_state=[int(x) for x in np.linspace(start=0, stop=100, num = 1)]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'max_leaf_nodes' : max_leaf_nodes,
               #'n_jobs':n_jobs,
               'bootstrap': bootstrap,
                #'random_state':random_state,
               #'criterion':criterion,
              }
pprint(random_grid)
from sklearn.model_selection import GridSearchCV
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 85,90, 100],
    'max_features': [1,2],
    'min_samples_leaf': [4, 5,6],
    'min_samples_split': [5,8, 10],
    'n_estimators': [100,120,140]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(X, y)
grid_search.best_params_
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100,random_state=0, cv = 3,  verbose=2, n_jobs = -1)
                               #random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X, y)
rf_random.best_params_

base_model = RandomForestRegressor(n_estimators = 220, min_samples_split=12,  min_samples_leaf=11, max_depth=120, max_features=1,
                                   bootstrap=True,
                                   random_state = 42)
base_model.fit(X,y)
z=base_model.predict(testx)
z
'''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)'''
'''from sklearn.ensemble import RandomForestClassifie
clf1=RandomForestClassifier(n_estimators=10,random_state=0)'''
#clf1.fit(X, y)
#p=clf1.predict(testx)
#du=le11.inverse_transform(p)
#du
id=test['Employee_ID'].values
data=pd.DataFrame({'Employee_ID':id,'Attrition_rate':z})
data.to_csv('submission.csv',index=False)
data
