import pandas as pd



train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')
train_df.head(20)
test_df.head(20)
train_df.shape
train_df.isnull().sum()
train_df.drop(["Name","Ticket","Cabin"], axis = 1, inplace = True)

train_df["Age"].fillna((train_df["Age"].mean()), inplace=True)
train_df = train_df.dropna()
train_df.isnull().sum()
train_df.head(20)
train_df["Survived"].value_counts().plot(kind="bar")
train_df.plot(kind = "scatter", x = "Survived", y="Age")
df = []

df.append(list(train_df[train_df["Pclass"]==1]["Survived"].value_counts().sort_index()))

df.append(list(train_df[train_df["Pclass"]==2]["Survived"].value_counts().sort_index()))

df.append(list(train_df[train_df["Pclass"]==3]["Survived"].value_counts().sort_index()))

df.append(list(train_df[train_df["Sex"]=="male"]["Survived"].value_counts().sort_index()))

df.append(list(train_df[train_df["Sex"]=="female"]["Survived"].value_counts().sort_index()))

df.append(list(train_df[train_df["Embarked"]=="S"]["Survived"].value_counts().sort_index()))

df.append(list(train_df[train_df["Embarked"]=="C"]["Survived"].value_counts().sort_index()))

df.append(list(train_df[train_df["Embarked"]=="Q"]["Survived"].value_counts().sort_index()))



labels = ["class 1","class 2","class 3","male","female","S","C","Q"]

notsurvived = [sub[0] for sub in df]

survived = [sub[1] for sub in df]



import numpy as np

from matplotlib import pyplot as plt



x = np.arange(len(labels))

width = 0.35



fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, notsurvived, width, label='Died')

rects2 = ax.bar(x + width/2, survived, width, label='Survived')



ax.set_ylabel('Number of people')

ax.set_title('Survivors')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()



fig.tight_layout()



plt.show()
train_df[train_df["Survived"]==1]["Parch"].value_counts().sort_index().plot(kind = "bar", title = "Survived")

train_df[train_df["Survived"]==0]["Parch"].value_counts().sort_index().plot(kind = "bar", title = "Died")

train_df[train_df["Survived"]==1]["SibSp"].value_counts().sort_index().plot(kind = "bar",title = "Survived")

train_df[train_df["Survived"]==0]["SibSp"].value_counts().sort_index().plot(kind = "bar",title = "Died")

train_df[train_df["Survived"]==1]["Fare"].value_counts().sort_index().plot( title = "Survived")

train_df[train_df["Survived"]==0]["Fare"].value_counts().sort_index().plot( title = "Died")

train_df[(train_df["Sex"]=="female")&(train_df["Pclass"]==1)]["Survived"].value_counts().plot(kind="pie")

train_df[(train_df["Sex"]=="male")&(train_df["Pclass"]==3)]["Survived"].value_counts().plot(kind="pie")

train_df.dtypes

print(train_df["Sex"].unique())

print(train_df["Embarked"].unique())
sex_mapping = {"male": 0, "female": 1}

train_df['Sex'] = train_df['Sex'].map(sex_mapping)

print(train_df["Sex"].unique())
embarked_mapping = {"S":0,"C":1,"Q":2}

train_df['Embarked'] = train_df['Embarked'].map(embarked_mapping)

print(train_df["Embarked"].unique())
test_df.head(10)

test_df["Age"].fillna((test_df["Age"].mean()), inplace=True)

test_df["Fare"].fillna((test_df["Fare"].mean()), inplace=True)
test_df.isnull().sum()
test_df.drop(["Name","Ticket","Cabin"], axis = 1, inplace = True)

test_df['Sex'] = test_df['Sex'].map(sex_mapping)

test_df['Embarked'] = test_df['Embarked'].map(embarked_mapping)
test_df.head(10)
y_try = train_df["Survived"]

X_try = train_df.drop("Survived",axis = 1)
from sklearn.svm import SVC

model = SVC(kernel="linear",random_state=0)

model.fit(X_try,y_try)
prediction = model.predict(test_df)
output = pd.DataFrame({'PassengerId': test_df["PassengerId"], 'Survived': prediction})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")