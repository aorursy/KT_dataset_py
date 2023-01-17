import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set_style('darkgrid')



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split  

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report,confusion_matrix
def family_member(col):

  sibsp=col[0]

  parch=col[1]

  sum_member = sibsp + parch + 1

  return sum_member

 
train=pd.read_csv('../input/titanic/train.csv')

test=pd.read_csv('../input/titanic/test.csv')
all_data = pd.concat([train.drop(['Survived'],axis=1),test])

combine_data = [train,test]
all_data.info()
vacant_data = all_data.isnull().sum().sort_values(ascending=False)

total_data = all_data.isnull().count().sort_values(ascending=False)

vacant_percent = (vacant_data/total_data)

missing_df = pd.concat([vacant_data,vacant_percent],axis=1,keys=['Total','Percent'])

missing_df.head()
sns.distplot(train['Age'],bins=20)

#hence there are no missing values with -1, or any other value
for dataset in combine_data:

  dataset['family_members']= dataset['SibSp']+dataset['Parch']+1



train.head()
plt.figure(figsize=(10,6))

plt.subplot(231)

sns.barplot('Pclass','Survived',data=train,palette='deep')

plt.subplot(232)

sns.barplot('Sex','Survived',data=train,palette='deep')

plt.subplot(233)

sns.barplot('SibSp','Survived',data=train,palette='deep')

plt.subplot(234)

sns.barplot('Parch','Survived',data=train,palette='deep')

plt.subplot(235)

sns.barplot('Embarked','Survived',data=train,palette='deep')

plt.subplot(236)

sns.barplot('family_members','Survived',data=train,palette='deep')

plt.tight_layout()
plt.figure(figsize=(10,6))

plt.subplot(221)

sns.countplot('Pclass',data=train,hue='Sex',palette='deep')

plt.subplot(222)

sns.barplot('Pclass','Survived',data=train,hue='Sex',palette='deep')

plt.tight_layout()



sns.countplot('Embarked',hue='Pclass',data=train,palette='deep',alpha=0.7)

tab = pd.crosstab(train['Pclass'],train['Embarked'])

print(tab,'\n')



g=sns.FacetGrid(train,col='Survived',hue='Sex',palette='deep')

g.map(plt.hist,'Age',bins=20,alpha=0.5)

# age till 5 have more survival rate. 

# age group of 20-30 didnt survive more.

#as age increases survival rate decreases.
grid=sns.FacetGrid(train,row='Survived',col='Pclass',hue='Sex',palette='deep')

grid.map(plt.hist,'Age',bins=20,alpha=0.5)

plt.legend()

#   CONFORM : that females in all classes have more survival rate

#   MALES from Pclass=3 are mostly didnt survived
grid=sns.FacetGrid(train,col='Embarked')

grid.map(sns.pointplot,'Pclass','Survived','Sex',palette='deep')

plt.legend()
grid=sns.FacetGrid(train,row='Survived',col='Embarked')

grid.map(sns.boxplot,'Sex','Fare',palette='deep')

plt.tight_layout()

# Generally survivals have paid more which correlates to they majorly belongs to Pclass=1.
#Variation of Fare with Embarked Placess

plt.figure(figsize=(14 ,6))

plt.subplot(121)

sns.boxplot(y='Fare',x='Embarked',data=train,hue='Sex')

plt.subplot(122)

sns.boxplot(x='Embarked',y='Fare',data=train,hue='Pclass')
# working on Title 

for dataset in combine_data:

  dataset['Title']=dataset['Name'].str.extract('([A-Za-z]+)\.')



#obtaining common title 

common_title = []

for title in train['Title'].unique():

  if train[train['Title'] == title]['Title'].count() > 20:

    common_title.append(title)

print(common_title) 



#using common list to  identify rare titles

def id_rare(cols):

  if cols in common_title:

    return cols 

  else:

    return 'Rare'

for dataset in combine_data:

  dataset['Title']=dataset['Title'].apply(id_rare)  



#mapping categorical into discrete numerical 

title_mapping = {'Mrs':1,'Miss':2,'Master':3,'Mr':4,'Rare':5}

for dataset in combine_data:

  dataset['Title']=dataset['Title'].map(title_mapping)



train.head()
map_sex = {'male':0,'female':1}

for dataset in combine_data:

 dataset['Sex']=dataset['Sex'].map(map_sex)
plt.figure(figsize=(12,6))

plt.subplot(121)

sns.boxplot(x='Pclass',y='Age',data=train,hue='Sex',palette='deep')

plt.subplot(122)

sns.boxplot(x='Pclass',y='Age',data=test,hue='Sex',palette='deep')
for dataset in combine_data:

  dataset['Age']=dataset.groupby(['Sex','Pclass'])['Age'].apply(lambda x : x.fillna(x.median()))
train['Embarked']=train["Embarked"].fillna('C')
fill_fare = test[(test['Pclass']==3)&(test['Sex']==0)]['Fare'].median()

test['Fare']=test['Fare'].fillna(fill_fare)
#reduces complexity

for dataset in combine_data:

  dataset['Age']=dataset['Age'].astype(int)



for dataset in combine_data:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4



for dataset in combine_data:

  dataset['is_alone']=0

  dataset.loc[dataset['family_members']==1,'is_alone'] = 1
map_embarked = {'S':0,'C':1,'Q':2}

for dataset in combine_data:

  dataset['Embarked']=dataset['Embarked'].map(map_embarked)



train.head(2)
train = train.drop(['PassengerId','Name','Ticket','Fare','Cabin','SibSp','Parch'],axis=1)

test = test.drop(['PassengerId','Name','Ticket','Fare','Cabin','SibSp','Parch'],axis=1) 
X=train.drop('Survived',axis=1)

y=train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=100)
log_model=LogisticRegression()

log_model.fit(X_train,y_train)

log_model_pred = log_model.predict(X_test)

acc_log_model = round(log_model.score(X_train,y_train)*100,2)

print('Accuracy of Logestic Model:',acc_log_model)

print('--*'*10)

print(classification_report(y_test,log_model_pred))

print('--*'*10)

print(confusion_matrix(y_test,log_model_pred))
tree_model=DecisionTreeClassifier()

tree_model.fit(X_train,y_train)

tree_model_pred = tree_model.predict(X_test)

acc_tree_model = round(tree_model.score(X_train,y_train)*100,2)

print('Accuracy of tree Model:',acc_tree_model)

print('--*'*10)

print(classification_report(y_test,tree_model_pred))

print('--*'*10)

print(confusion_matrix(y_test,tree_model_pred))
rf_model=RandomForestClassifier()

rf_model.fit(X_train,y_train)

rf_model_pred = rf_model.predict(X_test)

acc_rf_model = round(rf_model.score(X_train,y_train)*100,2)

print('Accuracy of Random Foresrt Model:',acc_rf_model)

print('--*'*10)

print(classification_report(y_test,rf_model_pred))

print('--*'*10)

print(confusion_matrix(y_test,rf_model_pred))
#training X,y in all of our model.

final_log = LogisticRegression()

final_log.fit(X,y)

final_log_pred = final_log.predict(test)



final_tree_model=DecisionTreeClassifier()

final_tree_model.fit(X,y)

final_tree_model_pred = final_tree_model.predict(test)



final_rf=RandomForestClassifier()

final_rf.fit(X,y)

final_rf_pred = final_rf.predict(test)

result= pd.read_csv('../input/titanic/gender_submission.csv')

result=result.set_index('PassengerId')

result.head(3)
result['Survived']= final_log_pred

result.to_csv('output_log.csv')



result['Survived']= final_tree_model_pred

result.to_csv('output_tree.csv')



result['Survived']= final_rf_pred

result.to_csv('output_rf.csv')
