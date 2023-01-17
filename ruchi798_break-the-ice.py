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

import pandas_profiling



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, roc_curve,auc, confusion_matrix, classification_report

import sklearn.metrics as metrics
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
train_data.describe()
train_data.dtypes
profile = pandas_profiling.ProfileReport(train_data)

profile
train_data.drop(["Cabin","Ticket", "Name", "PassengerId"], axis=1, inplace=True)

test_data.drop(["Cabin","Ticket", "Name", "PassengerId"], axis=1, inplace=True)



train_data["Age"].fillna(train_data["Age"].median(skipna=True), inplace=True)

test_data["Age"].fillna(test_data["Age"].median(skipna=True), inplace=True)





test_data["Fare"].fillna(test_data["Fare"].median(skipna=True), inplace=True)



train_data["Embarked"].fillna(train_data['Embarked'].value_counts().idxmax(), inplace=True)

test_data["Embarked"].fillna(test_data['Embarked'].value_counts().idxmax(), inplace=True)
gender = {'male': 0, 'female': 1}

train_data.Sex = [gender[item] for item in train_data.Sex] 

test_data.Sex = [gender[item] for item in test_data.Sex] 



embarked = {'S': 0, 'C': 1, 'Q':2}

train_data.Embarked = [embarked[item] for item in train_data.Embarked] 

test_data.Embarked = [embarked[item] for item in test_data.Embarked] 
train_data.dtypes
def plot_bar(df, feat_x, feat_y, normalize=True):

    """ Plot with vertical bars of the requested dataframe and features"""

    

    ct = pd.crosstab(df[feat_x], df[feat_y])

    if normalize == True:

        ct = ct.div(ct.sum(axis=1), axis=0)

    return ct.plot(kind='bar', stacked=True)
plot_bar(train_data, 'Pclass', 'Survived')

plt.show()
plot_bar(train_data, 'SibSp', 'Survived')

plt.show()
plot_bar(train_data, 'Parch', 'Survived')

plt.show()
plot_bar(train_data, 'Sex', 'Survived')

plt.show()
plot_bar(train_data, 'Embarked', 'Survived')

plt.show()
expected_values = train_data["Survived"]

train_data.drop("Survived", axis=1, inplace=True)
X = train_data.values

y = expected_values.values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                       criterion='gini', max_depth=4, max_features='auto',

                       max_leaf_nodes=5, max_samples=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=15,

                       min_weight_fraction_leaf=0.0, n_estimators=350,

                       n_jobs=None, oob_score=True, random_state=1, verbose=0,

                       warm_start=False)
model.fit(X_train, y_train)



y_pred_train = model.predict(X_train)

y_pred_test = model.predict(X_test)



print("Training accuracy: ", accuracy_score(y_train, y_pred_train))

print("Testing accuracy: ", accuracy_score(y_test, y_pred_test))

print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, y_pred_test))



fpr, tpr, _ = roc_curve(y_test, y_pred_test)

roc_auc = auc(fpr, tpr)

print("\nROC AUC on evaluation set",roc_auc )



plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()
model.fit(train_data, expected_values)

print("%.4f" % model.oob_score_)
passenger_IDs = pd.read_csv("/kaggle/input/titanic/test.csv")[["PassengerId"]].values

preds = model.predict(test_data.values)

preds
df = {'PassengerId': passenger_IDs.ravel(), 'Survived': preds}

df_predictions = pd.DataFrame(df).set_index(['PassengerId'])

df_predictions.head(10)
df_predictions.to_csv('/kaggle/working/Predictions.csv')