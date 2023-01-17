import pandas as pd

train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')
# 删除 Ticket、Cabin 和 Name

# delete Ticket, Cabin, Name

train_df = train_df.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin', 'Name'], axis=1)



combine = [train_df, test_df]



# sex 映射为数字

# map Sex to 1 and 0

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0}).astype(int)



# 用整体的年龄平均值填充缺失的年龄

# fill missing Ages with mean value

age_mean = (train_df['Age'].sum() + test_df['Age'].sum()) / (train_df['Age'].count() + test_df['Age'].count())

for dataset in combine:

    dataset['Age'].fillna(age_mean, inplace=True)



# 使用中位数填充 Fare 的缺失值

# fill missing Fares

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)



# 用数量最多的一个登船城市填充 Embarked 的缺失值

# fill missing Embarked

freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)



# Embarked 映射为数字

# map Embarked

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test_df.head()
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
Submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],

                            'Survived': Y_pred })

Submission.to_csv('submission.csv', index=False)