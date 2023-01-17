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
df_tr = pd.read_csv("/kaggle/input/titanic/train.csv")

df_te = pd.read_csv("/kaggle/input/titanic/test.csv")

df_gs = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

df_tgse =pd.concat([df_gs["Survived"],df_te] , axis =1)

dataset =pd.concat([df_tr,df_tgse], ignore_index = True)
Xtr = dataset.loc[:,["Sex","Age","SibSp","Parch"]]

Xtr.dropna(axis=0,inplace = True)

dftr = Xtr.groupby(["Parch","Sex"])["Age"].median()

df_train=dataset.copy()

df_train["Age"].fillna(0 ,inplace = True)
for i in range(len(df_train)):

    for j in range(0,7):

        if df_train["Age"][i] == 0  and df_train["Parch"][i] == j:

            df_train["Age"][i] = dftr[j][0]
df_train.isnull().sum()
new = df_train["Name"].str.split(",", n = 1, expand = True) 

new_1 =new[1].str.split(".", n = 1, expand = True) 

new_2 = df_train["Cabin"].str.extract('([a-zA-Z]+)([^a-zA-Z]+)', expand=True)
new_2
df_train["Name"] =new

df_train["Last Name"] =new_1[1]

df_train["Title"]=new_1[0]

df_train["Cabin ID"] =new_2[0]
df_train["Cabin ID"].fillna("Q" ,inplace = True)

df_train["Embarked"].fillna("S" ,inplace = True)
df_train["Pclass"].replace({1 : "first",

     2: "second",

     3: "third"},inplace = True)
df_train["SibSp"].replace({0 :"brother",

       1: "sister",

       2:"stepbrother",

       3:"stepsister",

       4:"husband",

       5:"wife",

       8:"ignored"},inplace = True)
df_train["Parch"].replace({0 :"daughter",

       1: "son",

       2:"stepdaughter",

       3:"stepson",

       4:"mother",

       5:"father",

       6:"nanny",

       9:"unknowns"},inplace = True)
df_train["Parch"]
dms_tr = pd.get_dummies(df_train[['Pclass','Sex', 

                               'SibSp',"Parch",

                               "Embarked","Title",

                               "Cabin ID"]])
X_tr = df_train.drop(['Pclass','Sex', 

                               'SibSp',"Parch",

                               "Embarked","Title",

                               "Cabin ID"], axis=1)
Xtrain = pd.concat([X_tr,dms_tr], axis = 1)
final_tr = Xtrain.iloc[0:891,:]

final_te = Xtrain.iloc[891:,:]
X_train = final_tr.drop(["PassengerId","Survived",

                       "Name",

                       "Ticket",

                       "Fare","Cabin","Last Name"],axis =1)

X_test = final_te.drop(["PassengerId","Survived",

                       "Name",

                       "Ticket",

                       "Fare","Cabin","Last Name"],axis =1)

y_train = final_tr["Survived"]

y_test = final_te["Survived"]
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report



from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

lgbm = LGBMClassifier()

lgbm_params = {"learning_rate": [0.001, 0.01, 0.1],

              "n_estimators": [200, 500, 100],

              "max_depth":[1,2,35,8]}

lgbm_cv_model = GridSearchCV(lgbm,lgbm_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)

lgbm_tuned = LGBMClassifier(learning_rate= lgbm_cv_model.best_params_['learning_rate'], 

                            max_depth= lgbm_cv_model.best_params_['max_depth'], 

                            n_estimators= lgbm_cv_model.best_params_['n_estimators']).fit(X_train, y_train)

y_pred = lgbm_tuned.predict(X_test)

accuracy_score(y_test, y_pred)
output = pd.DataFrame({'PassengerId': df_te.PassengerId, 'Survived': y_pred})

output
output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")