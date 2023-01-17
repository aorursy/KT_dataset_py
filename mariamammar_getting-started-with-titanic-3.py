# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

#from sklearn.preprocessing import StandardScaler




# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

train_data.isnull().sum()
imputer = SimpleImputer(strategy="mean")
imputer.fit(train_data[['Age']])
train_data['Age'] = imputer.transform(train_data[['Age']])
train_data["Age"].value_counts(dropna=False)
imputer.statistics_

log_model = LogisticRegression()
X = train_data[["Parch","Age","SibSp","Fare","Pclass"]]
y = train_data["Survived"]

log_model.fit(X, y)
log_model.score(X,y)
#scaler.fit(train_data[['Age']])
#scaler.fit(train_data[['Fare']])

#train_data["Age"] = scaler.transform(train_data[['Age']])
#train_data["Fare"] = scaler.transform(train_data[['Fare']])
normalizer = MinMaxScaler(feature_range=(0, 1))

normalizer.fit(train_data[['Age']])
train_data['Age'] = normalizer.transform(train_data[['Age']])

normalizer.fit(train_data[['Fare']])
train_data['Fare'] = normalizer.transform(train_data[['Fare']])

train_data.head()
le = LabelEncoder()
le.fit(train_data['Sex'])
train_data['Sex_encoded'] = le.transform(train_data['Sex'])
train_data.head()

log_model.fit(X,y)

log_model.score(X,y)

print(log_model.predict([[0,25,2,10.50,3]]))
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
output