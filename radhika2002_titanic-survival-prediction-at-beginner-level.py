# data analysis and wrangling
import pandas as pd
import numpy as np

# data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Upload CSV files
from google.colab import files
uploaded = files.upload()
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
print("Dimensions of train: {}".format(train_df.shape))
print("Dimensions of test: {}".format(test_df.shape))
#Data Exploration
train_df.info()
train_df.describe(include='all')
train_df.head(10)
#From train dataset it is observed that it contains different types of data such as numeric data, categorical, continous, missing/NaN values, which need to be taken care of
train_df.columns.values 
sns.set(style="darkgrid")
plt.figure(figsize = (8, 5))
ax= sns.countplot(x='Survived', hue="Survived", data=train_df)

plt.figure(figsize = (8, 5))
ax= sns.countplot(x='Sex', hue="Survived", data=train_df)

#By using heatmap
group = train_df.groupby(['Pclass','Survived'])
pclass_survived = group.size().unstack()
# Heatmap - Color encoded 2D representation of data. 
sns.heatmap(pclass_survived, annot = True, fmt ="d")
# Violinplot Displays distribution of data  
# across all levels of a category. 
sns.violinplot(x ="Sex", y ="Age", hue ="Survived",  
data = train_df, split = True)
sns.catplot(x ='Embarked', hue ='Survived',  
kind ='count', col ='Pclass', data = train_df) 
#Sibsp + Parch + Family, make them into Family COlumn
all_df = [train_df, test_df]

for i in all_df:
  i['Family'] = i['SibSp'] + i['Parch'] + 1

# Factorplot for Family_Size 
sns.factorplot(x ='Family', y ='Survived', data = train_df)
all_df = [train_df, test_df]

for data in all_df:
  data['IsAlone'] = 0
  data.loc[data['Family'] == 1, 'IsAlone'] = 1
# Factorplot for Alone 
sns.factorplot(x ='IsAlone', y ='Survived', data = train_df)

#So we will fill whose empty rows before doing computation.

all_df = [train_df, test_df]

for data in all_df:
  data['Fare'] = data['Fare'].fillna(data['Fare'].median())

# Barplot - Shows approximate values based  
# on the height of bars. 
sns.barplot(x ='category_fare', y ='Survived',  data = train_df) 
print( train_df[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean() )
print( train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean() )
print( train_df[["Embarked","Survived"]].groupby(["Embarked"], as_index = False).mean() )
print( train_df[["Family","Survived"]].groupby(["Family"], as_index = False).mean() )
train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df['category_fare'] = pd.qcut(train_df['Fare'], 4)
print( train_df[["category_fare","Survived"]].groupby(["category_fare"], as_index = False).mean() )
train_df = train_df.drop(['Cabin'], axis = 1)
test_df = test_df.drop(['Cabin'], axis = 1 )
train_df = train_df.drop(['Ticket'], axis = 1)
test_df = test_df.drop(['Ticket'], axis = 1)
print("Southampton(S):")
southampton = train_df[train_df['Embarked'] == 'S'].shape[0]
print(southampton)

print("Cherbourg(C):")
cherbourg = train_df[train_df['Embarked'] == 'C'].shape[0]
print(cherbourg)

print("Queenstown(Q):")
queenstown = train_df[train_df['Embarked'] == 'Q'].shape[0]
print(queenstown)
train_df = train_df.fillna({"Embarked": "S"})
train_df.Embarked.isnull().sum()
all_df = [train_df, test_df]
titles = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4,'Rare':5}

for i in all_df:
  i['Title'] = i.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
  i['Title'] = i['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major', 'Rev','Sir','Jonkheer','Dona'], 'Rare')
  i['Title'] = i['Title'].replace('Mlle', 'Miss')
  i['Title'] = i['Title'].replace('Ms', 'Miss')
  i['Title'] = i['Title'].replace('Mme', 'Mrs')
  # convert titles into numbers
  i['Title'] = i['Title'].map(titles)
  # filling NaN with 0, to get safe
  i['Title'] = i['Title'].fillna(0)


train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

all_df = [train_df, test_df]

for i in all_df:
  i['Age'] = i['Age'].replace(np.NaN, i['Age'].mean())
train_df.Age.describe()
all_df = [train_df, test_df]

for data in all_df:
  data['Age'] = data['Age'].astype(int)
  data.loc[data['Age'] <= 16, 'Age']                        = 0
  data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
  data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
  data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
  data.loc[data['Age'] > 64, 'Age']                        = 4
#Let us see its distribution
train_df['Age'].value_counts()
#Mapping Fare
all_df = [train_df, test_df]

for i in all_df:
  i.loc[i['Fare'] <= 7.91, 'Fare'] = 0
  i.loc[(i['Fare'] > 7.91) & (i['Fare'] <= 14.45), 'Fare'] = 1
  i.loc[(i['Fare'] > 14.45) & (i['Fare'] <= 31), 'Fare'] = 2
  i.loc[i['Fare'] > 31, 'Fare'] = 3
  i['Fare'] = i['Fare'].astype(int)
train_df =train_df.drop(['category_fare'], axis =1)

#Convert Sex column to Numerics
#Mapping Sex Column
all_df = [train_df, test_df]
gender = {"male":0, "female":1}

for i in all_df:
  i["Sex"] = i["Sex"].map(gender).astype(int)
#Convert all Port names to Numeric Form
#Mapping Embarked
all_df = [train_df, test_df]
port = {'S':0, 'C':1, 'Q':2}

for i in all_df:
  i['Embarked'] = i['Embarked'].map(port)
data = [train_df, test_df]
for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']
train_df = train_df.drop(['SibSp','Parch'], axis = 1)
test_df = test_df.drop(['SibSp','Parch'], axis = 1)
X_train = train_df.drop(['PassengerId', 'Survived'], axis= 1)
y_train = train_df['Survived']
X_test = test_df.drop(['PassengerId'], axis = 1)
#Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)
print("Logistic Regression Accuracy:", acc_log)
#KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
print("KNN Accuracy:",acc_knn)
#Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
acc_dt = round(dt.score(X_train, y_train) * 100, 2)
print("Decision Tree Accuracy:", acc_dt)
#Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc_rf = round(rf.score(X_train, y_train) * 100, 2)
print("Random Forest Accuracy:", acc_rf)
models = pd.DataFrame({
    'Model' : ['Logistic Regression', 'KNN', 'Random Forest','Decision Tree'],
    'Score' : [acc_log, acc_knn, acc_dt, acc_rf]
})
models.sort_values(by = 'Score', ascending= False)
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, y_train, cv=10, scoring = 'accuracy')

print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(rf.feature_importances_,2)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(rf, X_train, y_train, cv=3)
confusion_matrix(y_train, predictions)
from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(y_train, predictions))
print("Recall:",recall_score(y_train, predictions))
from sklearn.metrics import f1_score
f1_score(y_train, predictions)

#Create csv file to save results
submission = pd.DataFrame({
    'PassengerId' : test_df['PassengerId'],
    'Survived': y_pred
})

submission.to_csv('submission.csv', index = False)
