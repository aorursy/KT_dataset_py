import pandas as pd # processing data

import matplotlib.pyplot as plt # Visualization

import numpy as np # Linear Algebra

import re # Regular Expression

import seaborn as sns # Visaulization

from sklearn.preprocessing import LabelEncoder # Encoding object to number

from sklearn.ensemble import RandomForestClassifier #Classification model

from sklearn.model_selection import train_test_split #split train and validation data
train_data=pd.read_csv("../input/titanic/train.csv")

test_data=pd.read_csv("../input/titanic/test.csv")
train_data.head(10)
test_data.head(10)
p = re.compile('[^\,.$\.]+')
print(p.findall(train_data.Name[0]))

print(p.findall(test_data.Name[0]))
train_title = pd.Series(train_data.Name.map(lambda x : p.findall(x)[1]))

test_title = pd.Series(test_data.Name.map(lambda x : p.findall(x)[1]))
train_title = pd.Series(list(map(lambda x : x.strip() ,train_title)))

test_title = pd.Series(list(map( lambda x : x.strip(),test_title)))
plt.figure(figsize=(20,13))

sns.barplot(x = train_title.value_counts().index,y=train_title.value_counts().values)

plt.title("The title of aboarded")

plt.xlabel("title")

plt.ylabel("number of aboarded")

plt.show()
plt.figure(figsize=(20,13))

sns.barplot(x = test_title.value_counts().index,y=test_title.value_counts().values)

plt.title("The title of aboarded")

plt.xlabel("title")

plt.ylabel("number of aboarded")

plt.show()
train_data['title'] = train_title

test_data['title'] = test_title
train_data.head(10)
test_data.head()
list_for_check_train = []

for i in range(len(train_data)):

    if train_data.title[i] in train_data.Name[i]:

        list_for_check_train.append(True)

    else:

        list_for_check_train.append(False)

np.sum(list_for_check_train)
relation = train_data.corr()

plt.figure(figsize=(16,9))

sns.heatmap(data=relation,annot=True,cmap='YlGnBu')

plt.show()
train_data.count()
train_data[train_data.Age.isna()].title.value_counts()
test_data[test_data.Age.isna()].title.value_counts()
train_data.groupby('title').mean().T['Mr'].Age.mean()
for title in train_data[train_data.Age.isna()].title.value_counts().index:

    mean_age = train_data.groupby('title').mean().T[title].Age

    mean_age_list = train_data[train_data.title == title].Age.fillna(mean_age)

    train_data.update(mean_age_list)

train_data.Age.isna().sum()
for title in test_data[test_data.Age.isna()].title.value_counts().index:

    mean_age = train_data.groupby('title').mean().T[title].Age

    mean_age_list = test_data[test_data.title == title].Age.fillna(mean_age)

    test_data.update(mean_age_list)

test_data.Age.isna().sum()
train_data.count()
test_data.count()
test_data.Fare = test_data.Fare.fillna(train_data.Fare.mean())
most_frequnt_embarked_value = train_data.Embarked.value_counts()

train_data.Embarked = train_data.Embarked.fillna(most_frequnt_embarked_value.index[0])
train_data = train_data.drop(['Cabin','Ticket','PassengerId','Name'], axis= 1)

test_data = test_data.drop(['Cabin','Ticket','PassengerId','Name'], axis= 1)
train_data.select_dtypes(include='object').T.index
test_data.select_dtypes(include='object').T.index
train_data.count()
test_data.count()
label_encoder = LabelEncoder()
for col in list(train_data.select_dtypes(include='object').T.index):

    print(col)
train_data.head()
test_data.head()
for col in train_data.select_dtypes(include='object').T.index:

    train_data[col] = label_encoder.fit_transform(train_data[col])

    test_data[col] = label_encoder.fit_transform(test_data[col])
X_train = train_data.drop('Survived',axis=1)

y_train = train_data.Survived
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,test_size =0.2, random_state =2045)
modeler = RandomForestClassifier(random_state=2045)
modeler.fit(X_train,y_train)

pred = modeler.predict(X_valid)
np.sum(pred == y_valid)/len(pred == y_valid)
final_pred = modeler.predict(test_data)
final_pred
PassengerId = pd.read_csv("../input/titanic/test.csv").PassengerId
final = pd.DataFrame({'PassengerId':PassengerId,'Survived':final_pred})
final.to_csv('submission_MJ.csv', index=False)