import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
train_df = pd.read_csv('../input/titanic/train.csv')#train set

test_df = pd.read_csv('../input/titanic/test.csv')#test set
print(train_df.shape)

print(test_df.shape)
train_df.head()
test_df.head()
train_df.describe()
train_df.describe(include='object')
#looking for the types 

train_df.dtypes
#distribution of target variable

sns.distplot(train_df['Survived'])
#visualize the number of surivival

sur_count = sns.catplot(kind='count',

                       x ='Survived',

                       data =train_df,

                       palette ='ch:.25',

                       edgecolor ='0.5')
#relationship between Survived and Gender

sns.catplot(y='Survived',x='Sex',data=train_df,kind='bar')
#visualize the relationship between survival, pclass and gender 

sns.catplot(x="Sex",y="Survived",hue="Pclass",data=train_df,kind="point")
#visualize the relationship between survival,gender and embarked

sns.catplot(x='Sex',y='Survived',hue='Embarked',data=train_df,kind='bar')
#visualize the distributions of Age

sns.distplot(train_df['Age'])
train_df.isnull().sum()
test_df.isnull().sum()
def age_cabin(data):

    data['Age'].fillna(data['Age'].median(),inplace=True)#using median for fill the missing values

    data['Cabin'].fillna('NA',inplace=True)#fill the missing values as NA

age_cabin(train_df)

age_cabin(test_df)
plt.figure(figsize=(6,4))

plt.subplot(1,1,1)

ax = sns.countplot(x='Embarked',data=train_df,)

ax.set(title='Embarked in Train data')
def embarked(data):

    data['Embarked'].fillna('S',inplace=True)# fill the mode value in the missing places

    

embarked(train_df)
test_df['Fare'].fillna(test_df['Fare'].median(),inplace=True)#using median for fill the missing values
def add_relative(data):

    data['relative'] = data['SibSp']+data['Parch']

add_relative(train_df)

add_relative(test_df)
sns.catplot(x='relative',y='Survived',data=train_df,kind='bar')
def sex_num(data):

    lb_sex = LabelEncoder()

    data['sex_code'] = lb_sex.fit_transform(data['Sex'])#female-0,male-1



sex_num(train_df)    

sex_num(test_df)
#let us grouping the age using binning method

def age_num(data):

    bin_names = ['1','2','3','4','5','6']

    bins = np.linspace(min(data['Age']),max(data['Age']),7)

    data['age_code'] = pd.cut(data['Age'],bins,labels=bin_names,include_lowest=True)



age_num(train_df)

age_num(test_df)
#let us grouping the fare using binning method

def fare_num(data):

    group_names = ['1','2','3','4','5','6','7','8','9','10']

    bins = np.linspace(min(data['Fare']),max(data['Fare']),11)

    data['fare_code'] = pd.cut(data['Fare'],bins,labels=group_names,include_lowest=True)



fare_num(train_df)

fare_num(test_df)
def embarked_num(data):

    lb_embark = LabelEncoder()

    data['embarked_code'] = lb_embark.fit_transform(data['Embarked'])

    

embarked_num(train_df)

embarked_num(test_df)
train_df['Cabin'].unique()
def cabin_replace(data):

    data['cabin_code'] = data['Cabin'].str.extract('([A-Z]+)')

    data['cabin_code'].replace('NA','H',inplace=True)

    data['cabin_code'].replace('T','H',inplace=True)

    

cabin_replace(train_df)

cabin_replace(test_df)
def cabin_num(data):

    cabin_lb = LabelEncoder()

    data['cabin_code'] = cabin_lb.fit_transform(data['cabin_code'])



cabin_num(train_df)

cabin_num(test_df)
def title(data):

    data['name_title'] = data['Name'].str.extract('([A-Za-z]+)\.')

    

title(train_df)

title(test_df)
train_df['name_title'].unique()
def name_replace(data):

    data['name_title'].replace('Master','Mr',inplace =True)

    data['name_title'].replace('Ms','Miss',inplace=True)

    data['name_title'].replace(['Don','Rev','Dr','Mme','Major','Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Other',inplace=True)



name_replace(train_df)

name_replace(test_df)
train_df['name_title'].unique()
def name_num(data):

    name_lb = LabelEncoder()

    data['name_title'] = name_lb.fit_transform(data['name_title'])



name_num(train_df)

name_num(test_df)
def remove_var(data):

    removing_var = data[['Name','Sex','Age','Ticket','Fare','Cabin','Embarked']]

    data.drop(removing_var,axis=1,inplace=True)



remove_var(train_df)

remove_var(test_df)
train_df.columns
X = train_df.drop(['Survived'], axis=1)

y = train_df['Survived']
X = preprocessing.StandardScaler().fit_transform(X)

X[0:2]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

print('Train set:',X_train.shape, y_train.shape)

print('Test set:',X_test.shape, y_test.shape)
knn_model = KNeighborsClassifier(n_neighbors = 7)

knn_model.fit(X_train,y_train)

knn_pred = knn_model.predict(X_test)



knn_accuracy = metrics.accuracy_score(y_test,knn_pred)

print('Test set accuracy:',knn_accuracy)
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(knn_model,X_test,y_test,cmap='PuBu_r')
tree_model = DecisionTreeClassifier(criterion = 'entropy',max_depth=4)

tree_model.fit(X_train,y_train)

y_pred = tree_model.predict(X_test)



tree_accuracy = metrics.accuracy_score(y_test,y_pred)

print("Decision tree's accuracy:",tree_accuracy)
plot_confusion_matrix(tree_model,X_test,y_test,cmap='ocean_r')
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

y_pred = logmodel.predict(X_test)



log_accuracy = metrics.accuracy_score(y_test,y_pred) 

print('Accuracy:',log_accuracy)
plot_confusion_matrix(logmodel,X_test,y_test,cmap='OrRd')
best_model = knn_model

best_model.fit(X,y)

prediction = best_model.predict(test_df)
Submission = pd.DataFrame({'PassengerId': test_df['PassengerId'],'Survived': prediction})

Submission.to_csv("SurvivalSubmission.csv", index= False)