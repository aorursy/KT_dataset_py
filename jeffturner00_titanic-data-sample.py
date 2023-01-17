
import numpy as np 
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
data =  pd.read_csv("/kaggle/input/titanic/train.csv",index_col=0)
data.shape
test = pd.read_csv("/kaggle/input/titanic/test.csv", index_col=0)
test.shape
output_filename = "/kaggle/input/titanic/gender_submission.csv"
data.head()
data.describe()
'-----UniqueValues-----', data.nunique(), '-----NullCounts-----', data.isnull().sum(), data.info()
plt.figure(figsize=(16,5))
sns.heatmap(data.isnull(), yticklabels=False,cbar=False)
data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived')
sns.heatmap(data[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
data.loc[ data['Sex'] == 'male'] = 0
data.loc[ data['Sex'] == 'female'] = 1

data.head(1)
test.loc[ test['Sex'] == 'male'] = 0
test.loc[ test['Sex'] == 'female'] = 1

test.head(1)
y_train = data["Survived"]
x_train = data.drop("Survived", axis=1)
x_test = test
'Y shape: ', y_train.shape, 'X shape: ', x_train.shape, 'Test shape: ', test.shape
drops = ['Ticket', 'Cabin', 'Embarked', 'Name' ]
x_train = x_train.drop(drops, axis=1)
x_train.head(1)
x_test = x_test.drop(drops, axis=1)
x_test.head(1)
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
acc_log = round(model.score(x_train, y_train) * 100, 2)
acc_log
