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
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
import re

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.metrics import roc_curve, roc_auc_score





from xgboost import XGBClassifier

from lightgbm import LGBMClassifier
def preprecess(data):

    data.Cabin.fillna("N", inplace = True)

    data.loc[data.Cabin.str[0] == "A", "Cabin"] = "A"

    data.loc[data.Cabin.str[0] == "B", "Cabin"] = "B"

    data.loc[data.Cabin.str[0] == "C", "Cabin"] = "C"

    data.loc[data.Cabin.str[0] == "D", "Cabin"] = "D"

    data.loc[data.Cabin.str[0] == "E", "Cabin"] = "E"

    data.loc[data.Cabin.str[0] == "F", "Cabin"] = "F"

    data.loc[data.Cabin.str[0] == "G", "Cabin"] = "G"

    data.loc[data.Cabin.str[0] == "T", "Cabin"] = "N"

    

    data["Sex"].replace("female", 1, inplace=True)

    data["Sex"].replace("male", 0, inplace=True)   

    

    data["Age"].fillna(data["Age"].median(),inplace=True)

    data["Fare"].fillna(data["Fare"].median(),inplace=True)

    

    data.Pclass.replace(1,"First",inplace=True)

    data.Pclass.replace(2,"Second",inplace=True)

    data.Pclass.replace(3,"Third",inplace=True)

    

    data = pd.get_dummies(data=data, columns=["Cabin","Embarked","Pclass"])

    

    return data

    

def age_detection(value):

    if value < 18.0:

        return "kid"

    elif value >= 18.0 and value <=45.0:

        return "adult"

    else:

        return "mature"

        

    

def group_title(data):

    data["Title"] = data["Name"].map(lambda x: x.split(",")[1].split(".")[0].strip(" "))

    data['Title'].replace('Master', "A", inplace=True)

    data['Title'].replace('Mr', "B", inplace=True)

    data['Title'].replace(['Ms','Mlle', 'Miss'], "C", inplace=True)

    data['Title'].replace(['Mme', 'Mrs'], "D", inplace=True)

    data['Title'].replace(['Dona', 'Lady', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'the'], "E", inplace=True)

    data = pd.get_dummies(data=data, columns=["Title"])

    return data 



def outlier(data):

    for column in ["Age", "Fare"]:

        for sex in data.Sex.unique():

            sex_data = data[data["Sex"] == sex]

            sex_column = sex_data[column]

        

            Q1 = np.percentile(sex_column,25)

            Q3 = np.percentile(sex_column,75)

            IQR = Q3 - Q1

            STEP = 1.5 * IQR

            MAX_BORDER = Q3 + STEP

            MIN_BORDER = Q1 - STEP

        

            data.loc[(data["Sex"] == sex) & (data[column] > MAX_BORDER), column] = MAX_BORDER

            data.loc[(data["Sex"] == sex) & (data[column] < MIN_BORDER), column] = MIN_BORDER       

    return data



def scaler(data):

    data["Age"] = StandardScaler().fit_transform(data[["Age"]])

    data["Fare"] = StandardScaler().fit_transform(data[["Fare"]])

    return data
train = outlier(train)

test = outlier(test)

train["Age_Detection"] = train["Age"].apply(age_detection)

test["Age_Detection"] = test["Age"].apply(age_detection)

train = pd.get_dummies(data = train, columns = ["Age_Detection"])

test = pd.get_dummies(data = test, columns = ["Age_Detection"])

train = preprecess(train)

train = group_title(train)

test = preprecess(test)

test = group_title(test)

train = scaler(train)

test = scaler(test)

train.drop(["Name","Ticket"],axis = 1 ,inplace = True)

test.drop(["Name","Ticket"],axis = 1 ,inplace = True)
train.drop("PassengerId", axis = 1, inplace=True)
y = train.Survived

X = train.drop("Survived", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 1845)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
xgb_params = {

    'n_estimators' : [100,300,500,700],

    'subsample' : [0.6, 0.8, 1.0],

    'max_depth' : [3,4,5,6],

    'learning_rate' : [0.1, 0.01, 0.02, 0.05],

    'min_samples_split' : [2,5,10]

    

}





xgb = XGBClassifier()



xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = 10, n_jobs = -1, verbose = 2)
xgb_cv_model.fit(X_train, y_train)
xgb_cv_model.best_params_
xgb = XGBClassifier(learning_rate = 0.1,

                    max_depth = 5,

                    n_estimators = 100,

                    subsample = 0.8)

xgb_tuned = xgb.fit(X_train,y_train)
y_pred = xgb_tuned.predict(X_test)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
xgb_roc_auc = roc_auc_score(y_test, xgb_tuned.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, xgb_tuned.predict_proba(X_test)[:,1])



plt.figure()

plt.plot(fpr, tpr, label = "AUC (area = %0.2f)"% xgb_roc_auc)

plt.plot([0,1],[0,1], 'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC")

plt.legend(loc="lower right", fontsize=16)

plt.show()
lgbm_params = {

    'n_estimators' : [100,300],

    'subsample' : [0.6, 0.8, 1.0],

    'max_depth' : [3,4,5,6],

    'learning_rate' : [0.1, 0.01, 0.02, 0.05],

    'min_child_samples' : [5,10,20]    

}





lgbm = LGBMClassifier()



lgbm_cv_model = GridSearchCV(lgbm, lgbm_params, cv=10, n_jobs=-1, verbose = 2)
lgbm_cv_model.fit(X_train,y_train)
lgbm_cv_model.best_params_
lgbm = LGBMClassifier(learning_rate = 0.1,

                    max_depth = 4,

                    min_child_samples = 10,

                    n_estimators = 100,

                    subsample = 0.6)

lgbm_mode = lgbm.fit(X_train,y_train)
y_pred = lgbm_mode.predict(X_test)
accuracy_score(y_test, y_pred)
print(classification_report(y_test,y_pred))
lgbm_roc_auc = roc_auc_score(y_test, lgbm_mode.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, lgbm_mode.predict_proba(X_test)[:,1])



plt.figure()

plt.plot(fpr, tpr, label = "AUC (area = %0.2f)"% lgbm_roc_auc)

plt.plot([0,1],[0,1], 'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC")

plt.legend(loc="lower right", fontsize=16)

plt.show()
test_id = test.PassengerId

test.drop("PassengerId", axis=1, inplace=True)
xgb = XGBClassifier(learning_rate = 0.1,

                    max_depth = 6,

                    n_estimators = 100,

                    subsample = 0.8)

xgb_model = xgb.fit(X,y)
test_pred = xgb_model.predict(test)
submission = pd.DataFrame({"PassengerId" : test_id, "Survived" : test_pred})
submission.to_csv("submission.csv",index=False)