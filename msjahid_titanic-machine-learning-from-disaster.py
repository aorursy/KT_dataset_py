from sklearn.ensemble import RandomForestClassifier

import pandas as pd
#load the data from the system
train_data = pd.read_csv("../input/titanic/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_of_women = sum(women)/len(women)


print("% of women who survived:", rate_of_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_of_men = sum(men)/len(men)


print("% of men who survived:", rate_of_men)
y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")