import pandas as pd

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

gender_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train_data.head()

train_data.tail()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex =='male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived",rate_men)
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

print("Your submission was successfully saved!")