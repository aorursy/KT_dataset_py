# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

from xgboost import XGBClassifier

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data_drop = pd.read_csv("/kaggle/input/titanic/train.csv")



train_data['Died'] = 1 - train_data['Survived']

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
train_data.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True)

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

plt.show()
train_data.groupby('Pclass').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True)

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

plt.show()
train_data.groupby('SibSp').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True)

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

plt.show()



plt.hist([train_data[train_data['Survived'] == 1]['SibSp'], train_data[train_data['Survived'] == 0]['SibSp']], bins = 30, label = ['Survived','Dead'])

plt.xlabel('SibSp')

plt.ylabel('Number of passengers')

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

plt.show()
train_data.groupby('Parch').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True)

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

plt.show()



plt.hist([train_data[train_data['Survived'] == 1]['Parch'], train_data[train_data['Survived'] == 0]['Parch']], bins = 30, label = ['Survived','Dead'])

plt.xlabel('Parch')

plt.ylabel('Number of passengers')

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

plt.show()
plt.hist([train_data[train_data['Survived'] == 1]['Fare'], train_data[train_data['Survived'] == 0]['Fare']], bins = 30, label = ['Survived','Dead'])

plt.xlabel('Fare')

plt.ylabel('Number of passengers')

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

plt.show()
plt.hist([train_data[train_data['Survived'] == 1]['Age'], train_data[train_data['Survived'] == 0]['Age']], bins = 30, label = ['Survived','Dead'])

plt.xlabel('Age')

plt.ylabel('Number of passengers')

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

plt.show()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women survived:", rate_women)



men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men survived:", rate_men)
train_data.describe()
test_data.describe()
test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].median())





train_data['FareGroup'] = pd.cut(train_data['Fare'],4)

print(train_data[['FareGroup', 'Survived']].groupby('FareGroup', as_index=False).mean().sort_values('Survived', ascending=False))



def group_fare(fare):

    if fare <= 128: return 0

    if fare > 128 and fare <= 256: return 1

    if fare > 256 and fare <= 384: return 2

    if fare > 384 : return 3

    

for i, row in train_data.iterrows():

    train_data.at[i,'Fare Group'] = int(group_fare(row["Fare"]))

    

# Same for test data

for i, row in test_data.iterrows():

    test_data.at[i,'Fare Group'] = int(group_fare(row["Fare"]))



train_data.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import accuracy_score

import statistics 



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch", "Age","Embarked","Fare Group"]



X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



imp = SimpleImputer()

imp_X = imp.fit_transform(X)

imp_X_test = imp.transform(X_test)



model1 = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 1)

model1.fit(imp_X,y)

predictions1 = model1.predict(imp_X_test)



model1_preds = cross_val_predict(model1, imp_X, y)

model1_acc = accuracy_score(y, model1_preds)

print("Random Forest Accuracy:", model1_acc)
model2 = GradientBoostingClassifier(random_state=42)

model2.fit(imp_X,y)

predictions2 = model2.predict(imp_X_test)



model2_preds = cross_val_predict(model2, imp_X, y)

model2_acc = accuracy_score(y, model2_preds)

print("Gradient Booster Accuracy:", model2_acc)
model3 = XGBClassifier(max_depth=3, n_estimators=1000, learning_rate=0.05)

model3.fit(imp_X, y)

predictions3 = model3.predict(imp_X_test)



model3_preds = cross_val_predict(model3, imp_X, y)

model3_acc = accuracy_score(y, model3_preds)

print("XGBoost Accuracy:", model3_acc)
model4 = SVC(random_state = 1)

model4.fit(imp_X, y)

predictions4 = model4.predict(imp_X_test)



model4_preds = cross_val_predict(model4, imp_X, y)

model4_acc = accuracy_score(y, model4_preds)

print("SVC Accuracy:", model4_acc)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions1})

output.to_csv('my_submission.csv', index=False)

print("Your submission was succesfully saved!")