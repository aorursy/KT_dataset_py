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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df_train = pd.read_csv("../input/titanic/train.csv")

df_test = pd.read_csv("../input/titanic/test.csv")
df_train.head()
df_train.isnull().sum()
df_train["Age"].fillna(df_train["Age"].mean(),inplace=True)
for i, col in enumerate(["SibSp","Parch"]):

    plt.figure(i)

    sns.catplot(x=col,y="Survived",data=df_train,kind='point',aspect=2)
df_train["Family_count"]=df_train["SibSp"]+df_train["Parch"]
df_train.drop(["SibSp","Parch","PassengerId"],axis=1,inplace=True)
df_train.head(10)
df_train.to_csv('../../../titanic_cleaned.csv',index=False)
df_train['Cabin_ind']= np.where(df_train['Cabin'].isnull(), 0, 1)
gender_num = {'male':0, 'female': 1}

df_train["Sex"] = df_train["Sex"].map(gender_num)
df_train.drop(['Name','Ticket','Cabin','Embarked'],axis=1,inplace=True)
df_train.head()
df_train.to_csv('../../../titanic_cleaned.csv',index=False)
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



titanic = pd.read_csv('../../../titanic_cleaned.csv')

titanic.head()
features = titanic.drop("Survived", axis=1)

labels= titanic["Survived"]



# Split training data into training and test set

train_X, test_X ,train_y, test_y = train_test_split(features, labels, test_size=0.4,random_state=42)

# Split training data into training and test set

test_X ,val_X, test_y, val_y = train_test_split(test_X, test_y, test_size=0.5,random_state=42)
for dataset in [train_y, val_y, test_y]:

    print(round(len(dataset) / len(labels),2))
# Initialize model

model = RandomForestClassifier()

# Fit data

model.fit(train_X, train_y)

# Calc accuracy on test

acc = accuracy_score(model.predict(test_X), test_y)

print("Validation accuracy for Random Forest Model: {:.6f}".format(acc))
# Calc accuracy on validation

acc = accuracy_score(model.predict(val_X), val_y)

print("Validation accuracy for Random Forest Model: {:.6f}".format(acc))
from sklearn.model_selection import cross_val_score
# Initialize model

model2 = RandomForestClassifier()



scores = cross_val_score(model2,train_X,train_y, cv=5)
scores
from sklearn.model_selection import GridSearchCV
def print_results(results):

    print("BEST PARAMS {}\n".format(results.best_params_))

    

    means = results.cv_results_['mean_test_score']

    stds = results.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, results.cv_results_['params']):

        print("{} (+/-{}) for {}".format(round(mean,3),round(std * 2,3),params))
model3 = RandomForestClassifier()



parameters = {'n_estimators': [5,50,100],

           'max_depth': [2, 10, 20, None]

           }



cv = GridSearchCV(model3, parameters, cv=5)

cv.fit(train_X, train_y)
print_results(cv)
rf1 = RandomForestClassifier(max_depth=10,n_estimators= 50)

rf1.fit(train_X, train_y)



rf2 = RandomForestClassifier(max_depth=10,n_estimators= 100)

rf2.fit(train_X, train_y)



rf3 = RandomForestClassifier(max_depth=None,n_estimators=100)

rf3.fit(train_X, train_y)
from sklearn.metrics import precision_score, recall_score
for mdl in (rf1,rf2,rf3):

    y_pred = mdl.predict(val_X)

    accuracy = round(accuracy_score(val_y,y_pred),3)

    precision = round(precision_score(val_y,y_pred),3)

    recall = round(recall_score(val_y,y_pred),3)

    print("MAX_DEPTH: {}, / # OF EST: {} -- A: {}, P: {}, R: {}".format(mdl.max_depth,

                                                                        mdl.n_estimators,

                                                                        accuracy,

                                                                        precision,

                                                                        recall))

    
y_pred = rf2.predict(test_X)

accuracy = round(accuracy_score(test_y,y_pred),3)

precision = round(precision_score(test_y,y_pred),3)

recall = round(recall_score(test_y,y_pred),3)

print("MAX_DEPTH: {}, / # OF EST: {} -- A: {}, P: {}, R: {}".format(rf2.max_depth,

                                                                        rf2.n_estimators,

                                                                        accuracy,

                                                                        precision,

                                                                        recall))
final_model = rf2



df_test.head()

print(df_test.isnull().sum())

print(df_test.shape)
df_test["Age"].fillna(df_test["Age"].mean(),inplace=True)
df_test["Family_count"]=df_test["SibSp"]+df_test["Parch"]

df_test["Sex"] = df_test["Sex"].map(gender_num)

df_test['Cabin_ind']= np.where(df_test['Cabin'].isnull(), 0, 1)

df_test.drop(['Name','Ticket','Cabin','Embarked'],axis=1,inplace=True)



passenger_id = df_test["PassengerId"]
df_test[df_test.isnull().any(axis=1)]
means = df_test.groupby('Pclass')['Fare'].mean()

means
df_test["Fare"].fillna(12.459678,inplace=True)
df_test.drop(["SibSp","Parch","PassengerId"],axis=1,inplace=True)
df_test.head()
final_y_pred = final_model.predict(df_test)

final_sub = pd.DataFrame(passenger_id)

final_sub['Survived'] = final_y_pred

final_sub.to_csv('finalsub.csv',index=False,)