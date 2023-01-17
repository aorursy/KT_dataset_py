import numpy as np
import pandas as pd

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r"/kaggle/input/titanic/train.csv")
df.head()
test_data = pd.read_csv(r"/kaggle/input/titanic/test.csv")
test_data.head()
total = df.isnull().sum().sort_values(ascending= False)
percent1 = df.isnull().sum()/df.isnull().count()*100
percent2 = (round(percent1,1)).sort_values(ascending= False)
missing_data = pd.concat([total, percent2], axis=1, keys= ['Total','%'])
missing_data.head(5)
df= df.drop("Cabin",axis =1)
data = [df,test_data]
for dataset in data:

    mean = df["Age"].mean()

    std = test_data["Age"].std()

    is_null = dataset["Age"].isnull().sum()

    rand_age=  np.random.randint(mean-std,mean+std, size= is_null)

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = df["Age"].astype(int)

    

    
df["Age"].isnull().sum()
common_value = 'S'
data = [df,test_data]
for dataset in data:

    dataset["Embarked"] = dataset["Embarked"].fillna(common_value)
df.info()
genders= {"male":0, "female":1}

data = [df, test_data]
for dataset in data:

    dataset["Sex"] = dataset["Sex"].map(genders)
data = [df, test_data]



for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)
ports = {"S":0, "C":1,"Q":2}

data = [df, test_data]
for dataset in data:

    dataset["Embarked"] = dataset["Embarked"].map(ports)
df.info()
data = [df, test_data]

for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
df.head()
Xtrain = df.drop(["PassengerId","Survived","Name","Ticket"], axis =1)
Ytrain = df["Survived"]
Xtest = test_data.drop(["PassengerId","Name","Ticket"], axis =1)
random_forest = RandomForestClassifier(n_estimators = 100)
Xtest = Xtest.drop("Cabin" , axis=1)
random_forest.fit(Xtrain,Ytrain)
Xtest.head()
Xtrain.head()
Y_prediction = random_forest.predict(Xtest)
random_forest.score(Xtrain, Ytrain)
acc_random_forest = round(random_forest.score(Xtrain, Ytrain) * 100, 2)
print(acc_random_forest)
output = pd.DataFrame({"PassengerId": test_data.PassengerId,"Survived": Y_prediction})
output.to_csv("my_submssion.csv", index = False)
print("Your submission was successufly saved!")