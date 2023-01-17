# linear algebra

import numpy as np 



# data processing

import pandas as pd 



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style
train_users = pd.read_csv('../input/titanic/train.csv')

test_users = pd.read_csv('../input/titanic/test.csv')

print("There were", train_users.shape[0], "observations in the training set and", test_users.shape[0], "in the test set.")

print("In total there were", train_users.shape[0] + test_users.shape[0], "observations.")
plt.figure(figsize=(12,10))

cor = train_users.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
train_users.head(10)
test_users.head(10)
train_users.isnull().sum()
Age_Survived=train_users[['Age','Survived']]

Age_Survived.groupby(['Age'])['Survived'].aggregate('count').reset_index().sort_values('Survived', ascending=False)
Sex_Survived=train_users[['Sex','Survived']]

Sex_Survived.groupby(['Sex'])['Survived'].aggregate('count').reset_index().sort_values('Survived', ascending=False)
def unique_counts(train_users):

   for i in train_users.columns:

       count = train_users[i].nunique()

       print(i, ": ", count)

unique_counts(train_users)
plt.figure(figsize=(12,6))

sns.distplot(train_users.Age.dropna(), rug=True)

sns.despine()
plt.figure(figsize=(12,6))

sns.countplot(x='Survived', data=train_users)

plt.xlabel('Survived')

plt.ylabel('Number of the Survived')

sns.despine()
plt.figure(figsize=(12,6))

sns.countplot(x='Sex', data=train_users)

plt.xlabel('Sex')

plt.ylabel('Number of Sex')

sns.despine()
plt.figure(figsize=(12,6))

sns.boxplot(y='Fare', x='Sex',data=train_users)

plt.xlabel('Sex')

plt.ylabel('Fare')

plt.title('Sex vs. Fare')

sns.despine()
plt.figure(figsize=(15,12))



plt.subplot(211)

g = sns.countplot(x="Age", data=train_users, hue='Sex', dodge=True)

g.set_title("Age Count Distribution by Sex", fontsize=20)

g.set_ylabel("Count",fontsize= 17)

g.set_xlabel("Age", fontsize=17)

sizes=[]

for p in g.patches:

    height = p.get_height()

    sizes.append(height)

g.set_xlim(20,50)

g.set_ylim(0, max(sizes)*1.15)

plt.show()
plt.figure(figsize=(15,12))



plt.subplot(211)

g = sns.countplot(x="Fare", data=train_users, hue='Sex', dodge=True)

g.set_title("Fare by Sex", fontsize=20)

g.set_ylabel("Count",fontsize= 17)

g.set_xlabel("Fare", fontsize=17)

sizes=[]

for p in g.patches:

    height = p.get_height()

    sizes.append(height)

g.set_xlim(30,50)

plt.show()
plt.figure(figsize=(15,12))



plt.subplot(211)

g = sns.countplot(x="Pclass", data=train_users, hue='Survived', dodge=True)

g.set_title("Pclass by Survived", fontsize=20)

g.set_ylabel("Count",fontsize= 17)

g.set_xlabel("Pclass", fontsize=17)

sizes=[]

for p in g.patches:

    height = p.get_height()

    sizes.append(height)

plt.show()
data = train_users.append(test_users, ignore_index = True, sort = True)
data.head()
data["Sex"] = data["Sex"].apply(lambda x: x[0])
data["Sex"] = data["Sex"].apply(lambda x: 1 if x[0] == "m" else 0)
data["Family"] = data["Parch"].apply(lambda x: 1 if x>0 else 0)

data["Family with siblings/spouse"] = data["SibSp"].apply(lambda x: 1 if x >0 else 0 )
data["Embarked"] = data.Embarked.fillna('S')
data["Fare"] = data.Fare.fillna(0)
data["Age"]= data["Age"].fillna(data["Age"].mean())
data["Cabin"] = data.Cabin.fillna('Unknown_Cabin')

data['Cabin'] = data['Cabin'].str[0]
data['Cabin'] = np.where((data.Pclass==1) & (data.Cabin=='U'),'C',

                                            np.where((data.Pclass==2) & (data.Cabin=='U'),'D',

                                                                        np.where((data.Pclass==3) & (data.Cabin=='U'),'G',

                                                                                                    np.where(data.Cabin=='T','C',data.Cabin))))
data["Ticket"] =data["Ticket"].apply(lambda x: x[0])
data_num = ["Age", "Fare"]

drop = ["Name","Parch","SibSp","Ticket"]

data = data.drop(drop, axis = 1)

data_cat = ["Embarked", "PassengerId",

            'Cabin',"Pclass","Sex","Family","Family with siblings/spouse","Survived"]
num_data = data[data_num]

cat_data = data[data_cat]
data = pd.concat([num_data,cat_data], axis = 1)

bins = [0,12,18,40,50,data.Age.max()]

labels = ['Child', 'Young Adult', 'Adult','Older Adult','Senior']

data["Age"] = pd.cut(data["Age"], bins, labels = labels)

data = pd.get_dummies(data)
data.head()
from sklearn.metrics import f1_score

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score
Results = pd.DataFrame({'Model': [],'Accuracy Score': [], 'Recall':[], 'F1score':[]})
from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(data[data.Survived.isnull()==False].drop('Survived',axis=1),data.Survived[data.Survived.isnull()==False],test_size=0.40, random_state=2019)
from xgboost.sklearn import XGBClassifier

model = XGBClassifier(learning_rate=0.001,n_estimators=2500,

                                max_depth=4, min_child_weight=0,

                                gamma=0, subsample=0.7,

                                colsample_bytree=0.7,

                                scale_pos_weight=1, seed=27,

                                reg_alpha=0.00006)

model.fit(trainX, trainY)

y_pred = model.predict(testX)

from sklearn.metrics import accuracy_score

res = pd.DataFrame({"Model":['XGBClassifier'],

                    "Accuracy Score": [accuracy_score(y_pred,testY)],

                   "Recall": [recall_score(testY, y_pred)],

                   "F1score": [f1_score(testY, y_pred)]})

Results = Results.append(res)

Results
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=4)

model.fit(trainX, trainY)

y_pred = model.predict(testX)

res = pd.DataFrame({"Model":['DecisionTreeClassifier'],

                    "Accuracy Score": [accuracy_score(y_pred,testY)],

                   "Recall": [recall_score(testY, y_pred)],

                   "F1score": [f1_score(testY, y_pred)]})

Results = Results.append(res)

Results
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=2500, max_depth=4)

model.fit(trainX, trainY)

y_pred = model.predict(testX)

from sklearn.metrics import accuracy_score

res = pd.DataFrame({"Model":['RandomForestClassifier'],

                    "Accuracy Score": [accuracy_score(y_pred,testY)],

                   "Recall": [recall_score(testY, y_pred)],

                   "F1score": [f1_score(testY, y_pred)]})

Results = Results.append(res)

Results
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

model.fit(trainX, trainY)

y_pred = model.predict(testX)

from sklearn.metrics import accuracy_score

res = pd.DataFrame({"Model":['KNeighborsClassifier'],

                    "Accuracy Score": [accuracy_score(y_pred,testY)],

                   "Recall": [recall_score(testY, y_pred)],

                   "F1score": [f1_score(testY, y_pred)]})

Results = Results.append(res)

Results
from sklearn.svm import SVC

model = SVC()

model.fit(trainX, trainY)

y_pred = model.predict(testX)

from sklearn.metrics import accuracy_score

res = pd.DataFrame({"Model":['SVC'],

                    "Accuracy Score": [accuracy_score(y_pred,testY)],

                   "Recall": [recall_score(testY, y_pred)],

                   "F1score": [f1_score(testY, y_pred)]})

Results = Results.append(res)

Results
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(trainX, trainY)

y_pred = model.predict(testX)

from sklearn.metrics import accuracy_score

res = pd.DataFrame({"Model":['LogisticRegression'],

                    "Accuracy Score": [accuracy_score(y_pred,testY)],

                   "Recall": [recall_score(testY, y_pred)],

                   "F1score": [f1_score(testY, y_pred)]})

Results = Results.append(res)

Results
from xgboost import XGBClassifier

classifier = XGBClassifier(colsample_bylevel= 0.9999999,

                    colsample_bytree = 0.9999999, 

                    gamma=0.99999,

                    max_depth= 5,

                    min_child_weight= 1,

                    n_estimators= 1000,

                    nthread= 4,

                    random_state= 2,

                    silent= True)

classifier.fit(trainX,trainY)

xgb_predict = classifier.predict(testX)

from sklearn.metrics import accuracy_score

res = pd.DataFrame({"Model":['XGBOOST'],

                    "Accuracy Score": [accuracy_score(y_pred,testY)],

                   "Recall": [recall_score(testY, y_pred)],

                   "F1score": [f1_score(testY, y_pred)]})

Results = Results.append(res)

Results
from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import train_test_split

trainX = data[data.Survived.isnull()==False].drop(['Survived'],axis=1)

trainY = data.Survived[data.Survived.isnull()==False]

testX = data[data.Survived.isnull()==True].drop(['Survived'],axis=1)

model = XGBClassifier(learning_rate=0.001,n_estimators=2500,

                                max_depth=4, min_child_weight=0,

                                gamma=0, subsample=0.7,

                                colsample_bytree=0.7,

                                scale_pos_weight=1, seed=27,

                                reg_alpha=0.00006)
model.fit(trainX, trainY)

test_users['Survived'] = model.predict(testX).astype(int)

test_users = test_users.reset_index()
solution = pd.DataFrame({"PassengerId":test_users.PassengerId, "Survived":test_users.Survived})

solution.to_csv("Solution.csv", index = False)