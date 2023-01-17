import numpy as np

import pandas as pd

import os

from fancyimpute import KNN

from sklearn.model_selection import train_test_split

from xgboost import XGBRFClassifier

from sklearn.metrics import mean_absolute_error



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
train_data.info()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
test_data.info()
train_data['IsMale'] = train_data['Sex'].apply(lambda x: 1 if x == 'male' else 0)

train_data.head()
test_data['IsMale'] = test_data['Sex'].apply(lambda x: 1 if x == 'male' else 0)

test_data.head()
y = train_data["Survived"]



features = ["Pclass", "IsMale", "SibSp", "Parch", "Age", "Fare"]

X = train_data[features]

X.tail()
X_filled = KNN(k=3).fit_transform(X)

X_filled[-5:]
X_filled = pd.DataFrame(data=X_filled,columns=['Pclass','IsMale','SibSp','Parch','Age','Fare'])

X_filled.head()
X_train, X_valid, y_train, y_valid = train_test_split(X_filled, y, random_state=1)



model = XGBRFClassifier(n_estimators=1000, max_depth=5, random_state=1, learning_rate=0.5)

model.fit(X_train, y_train,

         early_stopping_rounds=5,

         eval_set=[(X_valid, y_valid)],

         verbose=False)

predictions = model.predict(X_valid)

print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
X_test = test_data[features]

X_test_filled = KNN(k=3).fit_transform(X_test)

X_test_filled = pd.DataFrame(data=X_test_filled,columns=['Pclass','IsMale','SibSp','Parch','Age','Fare'])

X_test_filled
predictions_final = model.predict(X_test_filled)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_final})

output
output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")