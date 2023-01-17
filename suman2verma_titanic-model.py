# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

path = os.getcwd()

print("Working Directory:", path)

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_train.head()

df_train.info()
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_test.head()
print ("Count of people with different age groups")

df_train['Age_Category'] = pd.cut(df_train['Age'],

                        bins=[0,16,32,48,64,81])

plt.subplots(figsize=(10,10))



sns.countplot('Age_Category',hue='Survived',data=df_train, palette= 'husl')

plt.show()
print ("Survival Ratio for Men And Women","\n","0:Not Survived","\n","1:Survived")

plt.subplots(figsize=(10,10))

sns.countplot('Sex',hue='Survived',data=df_train, palette = "Set3")

plt.show()

y = df_train["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(df_train[features])  

X_test = pd.get_dummies(df_test[features])

print ("Feature set",df_test )
print ("Target Column", y)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score





# RandomForest

print ("Training model with Random Forest Classifier")

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)



clf = RandomForestClassifier()

clf.fit(X, y)



y_pred_random = model.predict(X_test)

acc_random_forest = round(clf.score(X, y) * 100, 2)





# Knn

from sklearn.neighbors import KNeighborsClassifier

print ("Training model with KNN Classifier")

clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X, y)

y_pred_knn = model.predict(X_test)

acc_knn = round(clf.score(X, y) * 100, 2)





#Support Vector Classifier

from sklearn.svm import SVC, LinearSVC

print ("Training model with Linear Support Vector Classifier")

clf = SVC()

clf.fit(X, y)

y_pred_svc = clf.predict(X_test)

acc_linear_svc = round(clf.score(X, y) * 100, 2)



#Logestic Regression

from sklearn.linear_model import LogisticRegression

print ("Training model with  LogisticRegression Classifier")

clf = LogisticRegression()

clf.fit(X, y)

y_pred_log_reg = clf.predict(X_test)

acc_log_reg = round(clf.score(X, y) * 100, 2)



models = pd.DataFrame({

    'Model': ['Logistic Regression', 'KNN', 'SVC', 'Random Forest'],

    'Score': [acc_log_reg,acc_knn, acc_linear_svc, acc_random_forest]})



models.sort_values(by='Score', ascending=False)
X_test
submission = pd.DataFrame({

        "PassengerId": df_test['PassengerId'],

        "Survived": y_pred_random

    })





submission.to_csv('titanic_prediction.csv', index=False)






