# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#import train and test data.
train=pd.read_csv('/kaggle/input/wwwkagglecomprat57kaggletitanic/train.csv')
test=pd.read_csv('/kaggle/input/wwwkagglecomprat57kaggletitanic/test.csv')
name=train.Name
train.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
train.head(20)
train.shape
train.isnull().sum()
import numpy as np
import pandas as pd
train['Age']=train['Age'].fillna(train['Age'].median())

train.set_index('PassengerId',inplace=True)
# new_index = ['PassengerId']
# train.reindex(new_index )
## get dummy variables for Column sex and embarked since they are categorical value.
train['Sex']=train['Sex'].replace(to_replace='male',value=1)
train['Sex']=train['Sex'].replace(to_replace='female',value=0)
#train['Embarked']=train['Embarked'].replace(to_replace='female',value=0)
#train['Embarked']=train['Embarked'].replace(to_replace='female',value=1)
#train['Embarked']=train['Embarked'].replace(to_replace='female',value=2)

#Mapping the data.
train['Fare'] = train['Fare'].astype(int)
train.loc[train.Fare<=7.91,'Fare']=0
train.loc[(train.Fare>7.91) &(train.Fare<=14.454),'Fare']=1
train.loc[(train.Fare>14.454)&(train.Fare<=31),'Fare']=2
train.loc[(train.Fare>31),'Fare']=3

train['Age']=train['Age'].astype(int)
train.loc[ train['Age'] <= 16, 'Age']= 0
train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1
train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2
train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3
train.loc[train['Age'] > 64, 'Age'] = 4
train.drop(['Ticket','Cabin','Name','Embarked'],axis=1,inplace=True)
train.head()
print(type(train.Age))
train.shape
train.head(50)
train.Survived.value_counts()/len(train)*100
train.describe()
train.groupby('Survived').mean()
train.groupby('Sex').mean()
train.corr()
plt.subplots(figsize=(15,8))
sns.heatmap(train.corr(),annot=True,cmap="BrBG")
plt.title("Correlations Amongst Features", fontsize=15)
plt.subplots(figsize = (15,8))
sns.barplot(x = "Sex", y = "Survived", data=train, edgecolor=(0,0,0), linewidth=2)
plt.title("Survived/Non-Survived Passengers Gender-wise Distribution", fontsize = 25)
labels = ['Female', 'Male']
plt.ylabel("% of passenger survived", fontsize = 15)
plt.xlabel("Gender",fontsize = 15)
plt.xticks(sorted(train.Sex.unique()), labels)

# 1 is for male and 0 is for female.
sns.set(style='darkgrid')
plt.subplots(figsize = (15,8))
ax=sns.countplot(x='Sex',data=train,hue='Survived',edgecolor=(0,0,0),linewidth=2)
train.shape
## Fixing title, xlabel and ylabel
plt.title('Passenger distribution of survived vs not-survived',fontsize=25)
plt.xlabel('Gender',fontsize=15)
plt.ylabel("# of Passenger Survived", fontsize = 15)
labels = ['Female', 'Male']
#Fixing xticks.
plt.xticks(sorted(train.Survived.unique()),labels)
## Fixing legends
leg = ax.get_legend()
leg.set_title('Survived')
legs=leg.texts
legs[0].set_text('No')
legs[1].set_text('Yes')
plt.subplots(figsize = (10,10))
ax=sns.countplot(x='Pclass',hue='Survived',data=train)
plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 25)
leg=ax.get_legend()
leg.set_title('Survival')
legs=leg.texts

legs[0].set_text('No')
legs[1].set_text("yes")
plt.subplots(figsize=(10,8))
ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Pclass'],shade=True,color='r',label='Not Survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Pclass'],shade=True,color='b',label='Survived' )

labels = ['First', 'Second', 'Third']
plt.xticks(sorted(train.Pclass.unique()),labels)


plt.subplots(figsize=(15,10))

ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Fare'],color='r',shade=True,label='Not Survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Fare'],color='b',shade=True,label='Survived' )
plt.title('Fare Distribution Survived vs Non Survived',fontsize=25)
plt.ylabel('Frequency of Passengers Survived',fontsize=20)
plt.xlabel('Fare',fontsize=20)
fig,axs=plt.subplots(figsize=(10,8))
sns.set_style(style='darkgrid')
sns.kdeplot(train.loc[(train['Survived']==0),'Age'],color='r',shade=True,label='Not Survived')
sns.kdeplot(train.loc[(train['Survived']==1),'Age'],color='b',shade=True,label='Survived')
train.dtypes
X=train.drop('Survived',axis=1)
y=train['Survived'].astype(int)
train.head(20)
from sklearn.model_selection import train_test_split

#x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state =0)

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
X_train, X_test = X.iloc[train_index], X.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]
##Now we fit our model
from xgboost import XGBClassifier
classifier = XGBClassifier(colsample_bylevel= 0.9,
                    colsample_bytree = 0.8, 
                    gamma=0.99,
                    max_depth= 5,
                    min_child_weight= 1,
                    n_estimators= 10,
                    nthread= 4,
                    random_state= 2,
                    silent= True)
classifier.fit(X_train,y_train)
classifier.score(X_test,y_test)
#test = test_data['Pclass','Sex','Age','SibSp','Parch','Fare']
test.drop(['Ticket','Cabin','Name','Embarked'],axis=1,inplace=True)
test['Age']=test['Age'].fillna(test['Age'].median())
test['Fare']=test['Fare'].fillna(test['Fare'].median())
test.set_index('PassengerId',inplace=True)
# new_index = ['PassengerId']
# test.reindex(new_index )
## get dummy variables for Column sex and embarked since they are categorical value.
test['Sex']=test['Sex'].replace(to_replace='male',value=1)
test['Sex']=test['Sex'].replace(to_replace='female',value=0)


#Mapping the data.
test['Fare'] = test['Fare'].astype(int)
test.loc[test.Fare<=7.91,'Fare']=0
test.loc[(test.Fare>7.91) &(test.Fare<=14.454),'Fare']=1
test.loc[(test.Fare>14.454)&(test.Fare<=31),'Fare']=2
test.loc[(test.Fare>31),'Fare']=3

test['Age']=test['Age'].astype(int)
test.loc[ test['Age'] <= 16, 'Age']= 0
test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 1
test.loc[(test['Age'] > 32) & (test['Age'] <= 48), 'Age'] = 2
test.loc[(test['Age'] > 48) & (test['Age'] <= 64), 'Age'] = 3
test.loc[test['Age'] > 64, 'Age'] = 4
test.head()
Result=classifier.predict(test)
print(Result)
print(len(Result))
output = pd.DataFrame({'PassengerId': test.index,'Survived': Result})
output.to_csv('submission2.csv', index=False)
output.head()
