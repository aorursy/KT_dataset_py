# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd 

train = pd.read_csv('../input/learn-together/train.csv')

test = pd.read_csv('../input/learn-together/test.csv')
train.isnull().sum()
train.head()
train.info()
#no of unique values in each feature

for column in list(train.columns):

    print ("{0:25} {1}".format(column, train[column].nunique()))
#  droping not so useful training columns 

dropable_attributes = ['Id','Soil_Type7','Soil_Type15','Cover_Type']

X = train.drop((dropable_attributes), axis =1)

y = train['Cover_Type']



# creating test-train set  

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, random_state=42, test_size=.20)

 
from sklearn.ensemble import RandomForestClassifier

# define

rf = RandomForestClassifier()

# train

rf.fit(X_train,y_train)

from sklearn.model_selection import cross_val_score

accuracy = cross_val_score(rf, X, y, cv=5, scoring='accuracy')

acc = accuracy.mean()

acc
#### random search cv 

# Note: this code block will take time  to execute

from sklearn.model_selection import RandomizedSearchCV

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



 

params = {

        "n_estimators": [200, 400, 600   ],

        "criterion" : ["entropy", "gini"],

        "max_depth" : [  20, 40, 60],

        "max_features" : [ .30, .50 , .70 ],    

        "bootstrap" : [True, False]

           }



rs = RandomizedSearchCV(rf, params, cv=3, scoring='accuracy',verbose=10)

rs.fit(X_train, y_train) 

rs.best_params_
rf_rs = RandomForestClassifier(n_estimators= 400,max_features =0.3,

                               max_depth =40,criterion ='entropy', bootstrap= False )

rf_rs.fit(X_train,y_train)

accuracy = cross_val_score(rf_rs , X, y, cv=5, scoring='accuracy')

acc = accuracy.mean()

acc
def create_submission_file( predictions, name):

    submission = pd.DataFrame()

    submission['ID'] = test['Id']     

    submission['Cover_Type'] = predictions

    submission.to_csv( name+'.csv',index=False, header= True)
testcopy = test.drop((['Id','Soil_Type7','Soil_Type15']), axis =1)

predictions = rf_rs.predict(testcopy) 

create_submission_file( predictions, 'out')