import numpy as np
import pandas as pd
titanic_data = pd.read_csv("../input/train.csv")
titanic_test_data = pd.read_csv("../input/test.csv")


y_all_data = titanic_data.Survived
X_all_data = titanic_data.loc[:, "Pclass":"Embarked"]
# NOTE: Cabin might be helpful, but most of it is NaN, so it was ignored.
X_all_data = X_all_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
X_all_data = pd.get_dummies(X_all_data) # Pandas one-hot encodes for us, convenient!
# Age has NaN values, so let's fill them with mean Non-NaN age value
mean_age = round(X_all_data.Age.mean())
X_all_data.Age = X_all_data.Age.fillna(mean_age)
from sklearn.model_selection import train_test_split
X_train,X_val, y_train,y_val = train_test_split(X_all_data, y_all_data, test_size=0.2, random_state=42)

print(X_train[0:2])
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_val)

from sklearn.metrics import accuracy_score
print("Accuracy for validation data is:")
print(str(round(accuracy_score(predictions, y_val)*100, 2)) + "%")
# Prepare test data in similar way to train, but keep passenger_ids
X_test_ids = titanic_test_data.PassengerId
X_test_data = titanic_test_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
X_test_data = pd.get_dummies(X_test_data)
mean_test_age = round(X_test_data.Age.mean())
X_test_data.Age = X_test_data.Age.fillna(mean_test_age)

# Test data has one NaN Fare value, apply the same operation as age, set it to mean of all others.
mean_test_fare = X_test_data.Fare.mean()
X_test_data.Fare = X_test_data.Fare.fillna(mean_test_fare)

test_predictions = model.predict(X_test_data)
test_predictions = pd.DataFrame({"PassengerId":X_test_ids, "Survived":test_predictions})
test_predictions.to_csv("result.csv", index=False)


