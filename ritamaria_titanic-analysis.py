#The data sets are given by kaggle --> https://www.kaggle.com/c/titanic/data



#Load the data



import pandas as pd



train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')
train_data.head(10)
train_data.shape
train_data.isnull().sum()
train_data.drop(["Name","Ticket","Cabin"], axis = 1, inplace = True)
#fill missing age with median age

train_data["Age"].fillna((train_data["Age"].mean()), inplace=True)
#remove the rows with missing values

train_data = train_data.dropna()
train_data.isnull().sum()
train_data.head(10)
train_data["Survived"].value_counts().plot(kind="bar", color = "orange")
train_data.plot(kind = "scatter", x = "Survived", y="Age", color = "olive")
data = []

data.append(list(train_data[train_data["Pclass"]==1]["Survived"].value_counts().sort_index()))

data.append(list(train_data[train_data["Pclass"]==2]["Survived"].value_counts().sort_index()))

data.append(list(train_data[train_data["Pclass"]==3]["Survived"].value_counts().sort_index()))

data.append(list(train_data[train_data["Sex"]=="male"]["Survived"].value_counts().sort_index()))

data.append(list(train_data[train_data["Sex"]=="female"]["Survived"].value_counts().sort_index()))

data.append(list(train_data[train_data["Embarked"]=="S"]["Survived"].value_counts().sort_index()))

data.append(list(train_data[train_data["Embarked"]=="C"]["Survived"].value_counts().sort_index()))

data.append(list(train_data[train_data["Embarked"]=="Q"]["Survived"].value_counts().sort_index()))



labels = ["class 1","class 2","class 3","male","female","S","C","Q"]

notsurvived = [sub[0] for sub in data]

survived = [sub[1] for sub in data]



import numpy as np

from matplotlib import pyplot as plt



x = np.arange(len(labels))

width = 0.35



fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, notsurvived, width, label='Died', color = "lightcoral")

rects2 = ax.bar(x + width/2, survived, width, label='Survived', color = "lightgreen")



ax.set_ylabel('Number of people')

ax.set_title('Survivors')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()



fig.tight_layout()



plt.show()
train_data[train_data["Survived"]==1]["Parch"].value_counts().sort_index().plot(kind = "bar", title = "Survived")
train_data[train_data["Survived"]==0]["Parch"].value_counts().sort_index().plot(kind = "bar", title = "Died")
train_data[train_data["Survived"]==1]["SibSp"].value_counts().sort_index().plot(kind = "bar", color = "purple",title = "Survived")
train_data[train_data["Survived"]==0]["SibSp"].value_counts().sort_index().plot(kind = "bar", color = "purple",title = "Died")
train_data[train_data["Survived"]==1]["Fare"].value_counts().sort_index().plot(color = "silver", title = "Survived")
train_data[train_data["Survived"]==0]["Fare"].value_counts().sort_index().plot(color = "silver", title = "Died")
train_data[(train_data["Sex"]=="female")&(train_data["Pclass"]==1)]["Survived"].value_counts().plot(kind="pie", colors = ["g","r"])
train_data[(train_data["Sex"]=="male")&(train_data["Pclass"]==3)]["Survived"].value_counts().plot(kind="pie", colors = ["r","g"])
#Look at the data types

train_data.dtypes
print(train_data["Sex"].unique())

print(train_data["Embarked"].unique())
sex_mapping = {"male": 0, "female": 1}

train_data['Sex'] = train_data['Sex'].map(sex_mapping)

print(train_data["Sex"].unique())
embarked_mapping = {"S":0,"C":1,"Q":2}

train_data['Embarked'] = train_data['Embarked'].map(embarked_mapping)

print(train_data["Embarked"].unique())
test_data.head()
test_data["Age"].fillna((test_data["Age"].mean()), inplace=True)

test_data["Fare"].fillna((test_data["Fare"].mean()), inplace=True)
test_data.isnull().sum()
test_data.drop(["Name","Ticket","Cabin"], axis = 1, inplace = True)
test_data['Sex'] = test_data['Sex'].map(sex_mapping)
test_data['Embarked'] = test_data['Embarked'].map(embarked_mapping)
test_data.head()
y_train = train_data["Survived"]

X_train = train_data.drop("Survived",axis = 1)
from sklearn.svm import SVC

model = SVC(kernel="linear",random_state=0)

model.fit(X_train,y_train)
prediction = model.predict(test_data)
output = pd.DataFrame({'PassengerId': test_data["PassengerId"], 'Survived': prediction})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
output.head()