# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv as csv
import xgboost as xgb


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv")
median_age = df_train['Age'].dropna().median()
if len(df_train.Age[ df_train.Age.isnull() ]) > 0:
    df_train.loc[ (df_train.Age.isnull()), 'Age'] = median_age
df_train.info()
df_train_x = df_train.drop(['PassengerId','Name','Survived','Cabin','Ticket','Embarked'], axis = 1)
df_train_x.info()
df_train_y = df_train[['Survived']]
df_train_y.info()
ohe = ['Sex']
for f in ohe:
    df_train_dummy = pd.get_dummies(df_train_x[f], prefix = f)
    df_train_x = df_train_x.drop([f], axis = 1)
    df_train_x = pd.concat((df_train_x, df_train_dummy), axis = 1)

# df_train_x.loc[df_train_x["Sex"] == "male", "Sex"] = 0
#df_train_x.loc[df_train_x["Sex"] == "female", "Sex"] = 1

#df_train_x["Embarked"] = df_train_x["Embarked"].fillna("S")

#df_train_x.loc[df_train_x["Embarked"] == "S", "Embarked"] = 0
#df_train_x.loc[df_train_x["Embarked"] == "C", "Embarked"] = 1
#df_train_x.loc[df_train_x["Embarked"] == "Q", "Embarked"] = 2   
#df_train_x.info()
from sklearn.ensemble import RandomForestClassifier
class_w = {'Pclass':2 ,'Sex':2 ,'Fare':2}

forest = RandomForestClassifier(n_jobs=-1, n_estimators=100, class_weight='balanced')
forest = forest.fit(df_train_x, df_train_y )
forest.score(df_train_x, df_train_y)
df_test = pd.read_csv("../input/test.csv")
ids = df_test['PassengerId'].values
df_test = df_test.drop(['PassengerId','Name','Cabin','Ticket','Embarked'], axis = 1)
df_test.info()
median_age = df_test['Age'].dropna().median()
if len(df_test.Age[ df_test.Age.isnull() ]) > 0:
    df_test.loc[ (df_test.Age.isnull()), 'Age'] = median_age
    
median_fare = df_test['Fare'].dropna().median()
if len(df_test.Fare[ df_test.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = df_test[ df_test.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        df_test.loc[ (df_test.Fare.isnull()) & (df_test.Pclass == f+1 ), 'Fare'] = median_fare[f]   

df_test.info()
ohe = ['Sex']
for f in ohe:
    df_test_dummy = pd.get_dummies(df_test[f], prefix = f)
    df_test = df_test.drop([f], axis = 1)
    df_test = pd.concat((df_test, df_test_dummy), axis = 1)
    
#df_test.loc[df_test["Sex"] == "male", "Sex"] = 0
#df_test.loc[df_test["Sex"] == "female", "Sex"] = 1

#df_test["Embarked"] = df_test["Embarked"].fillna("S")

#df_test.loc[df_test["Embarked"] == "S", "Embarked"] = 0
#df_test.loc[df_test["Embarked"] == "C", "Embarked"] = 1
#df_test.loc[df_test["Embarked"] == "Q", "Embarked"] = 2  
df_test.info()
output = forest.predict(df_test)
gbm = xgb.XGBClassifier()
gbm = gbm.fit(df_train_x, df_train_y)
gbm.score(df_train_x, df_train_y)
output_gbm = gbm.predict(df_test)
output_csv = zip(ids, output)
output_gbm_csv = zip(ids, output_gbm)

predictions_file = open("myfirstforest.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(['PassengerId', 'Survived'])
open_file_object.writerows(output_gbm_csv)
predictions_file.close()
