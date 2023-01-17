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
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
correct = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train.head()
def intGender(row):
    if row['Sex']=="male":
        row['Sex']=0
        #print(row['Sex'])
    else:
        row['Sex']=1
        #print(row['Sex'])
    return row

train = train.apply(lambda row: intGender(row), axis = 1)
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(train["Survived"])
sns.countplot(train["Pclass"], hue=train['Survived'])
sns.countplot(train["SibSp"], hue=train['Survived'])
sns.countplot(train["Parch"], hue=train['Survived'])
sns.countplot(train["Embarked"], hue=train['Survived'])
sns.distplot(train["Fare"], bins=20)
sns.distplot(train["Age"], bins=20)
sns.countplot(train['Sex'], hue=train['Survived'])
sns.heatmap(train.corr(), annot=True)
for col in train.columns:
    print(col + " : " + str(train[~train[col].isna()].shape[0]/train.shape[0]))
train = train.drop(['Age', 'Cabin', 'Embarked', 'Name', 'Ticket', 'PassengerId'], axis = 1)
from sklearn.model_selection import train_test_split
import statsmodels.api as sa
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

def getVIF(X):
    v = pd.DataFrame()
    v["Features"] = X.columns
    v["VIF"] = [vif(X.values,i) for i in range(X.shape[1])]
    return v
X = train.drop("Survived", axis = 1)
y = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
model = sa.OLS(y_train, sa.add_constant(X_train))
res = model.fit()
res.summary()
res.rsquared_adj
model = sa.OLS(y_train, sa.add_constant(X_train.drop("Fare", axis = 1)))
res = model.fit()
res.summary()
model = sa.OLS(y_train, sa.add_constant(X_train.drop(["Fare","Parch"], axis = 1)))
res = model.fit()
res.summary()
getVIF(X_train.drop(["Fare","Parch"], axis = 1))
y_pred = pd.DataFrame()
y_pred["Predicted"] = res.predict(sa.add_constant(X_test.drop(["Fare", "Parch"], axis = 1)))
def approxClean(row):
    if row["Predicted"] >= 0.9:
        return 1
    else:
        return 0

y_pred["Approx"] = y_pred.apply(lambda row : approxClean(row), axis = 1)
y_pred["Actual"] = y_test
y_pred
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
accuracy_score(y_pred["Actual"], y_pred["Approx"])
testX = test[["Pclass", "Sex", "SibSp"]]

def intGender(row):
    if row['Sex']=="male":
        row['Sex']=0
        #print(row['Sex'])
    else:
        row['Sex']=1
        #print(row['Sex'])
    return row

testX = testX.apply(lambda row: intGender(row), axis = 1)
test["Survived_Pred"] = res.predict(sa.add_constant(testX))

def approxClean(row):
    if row["Survived_Pred"] >= 0.5:
        return 1
    else:
        return 0

test["Survived"] = test.apply(lambda row : approxClean(row), axis = 1)
test
test[["PassengerId","Survived"]].to_csv("submit.csv",index = True)
glm_binom = sa.GLM(y_train, sa.add_constant(X_train), family=sa.families.Binomial())
res = glm_binom.fit()
print(res.summary())
glm_binom = sa.GLM(y_train, sa.add_constant(X_train.drop("Fare", axis = 1)), family=sa.families.Binomial())
res = glm_binom.fit()
print(res.summary())
glm_binom = sa.GLM(y_train, sa.add_constant(X_train.drop(["Fare","Parch"], axis = 1)), family=sa.families.Binomial())
res = glm_binom.fit()
print(res.summary())
glm_binom = sa.GLM(y_train, sa.add_constant(X_train.drop(["Fare","Parch", "SibSp"], axis = 1)), family=sa.families.Binomial())
res = glm_binom.fit()
print(res.summary())
def approxClean(row, threshold):
    if row["Survived_Pred"] >= threshold:
        return 1
    else:
        return 0
y_pred = pd.DataFrame()

y_pred["Actual"] = y_test
y_pred["Survived_Pred"] = res.predict(sa.add_constant(X_test.drop(["Fare","Parch", "SibSp"], axis = 1)))
y_pred["Survived"] = y_pred.apply(lambda row : approxClean(row, 0.5), axis = 1)
accuracy_score(y_pred["Actual"], y_pred["Survived"])
confusion_matrix(y_true=y_pred["Actual"], y_pred=y_pred["Survived"])
testX = test[["Pclass", "Sex"]]

test["Survived_Pred"] = res.predict(sa.add_constant(testX))

test["Survived"] = test.apply(lambda row : approxClean(row, 0.5), axis = 1)
test[["PassengerId","Survived"]].to_csv("submitLogReg.csv",index = False)
