import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

print(train_data.head())

print(test_data.head())

print(train_data.shape)

print(test_data.shape)
train_data['Embarked'] = train_data['Embarked'].fillna('S')

#train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

train_data['Sex'] = train_data['Sex'].replace(['male','female'],[0,1])

train_data['Embarked'] = train_data['Embarked'].replace(['S','C','Q'],[0,1,2])

train_data = train_data.drop(['Name','PassengerId','Ticket','Cabin'],axis=1)



train_data.head()
train_data.isnull().sum()
from sklearn.model_selection import train_test_split

from sklearn import svm

train_data_y = train_data['Survived']

train_data_X = train_data.drop(['Survived'],axis=1)

#X_train,X_test,y_train,y_test = train_test_split(train_data_X,train_data_y,test_size=.3,random_state=25)

#svc_model = svm.SVC(C=100,gamma=0.01,kernel='rbf')

#svc_model.fit(train_data_X,train_data_y)

#sc = svc_model.score(train_data_X,train_data_y)

#print(sc)
test_data_pid = test_data['PassengerId']

test_data = test_data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

print(test_data.isnull().sum())
print(test_data.isnull().sum())

test_data['Sex'] = test_data['Sex'].replace(['male','female'],[0,1])

test_data['Embarked'] = test_data['Embarked'].replace(['S','C','Q'],[0,1,2])
plt.figure()

plt.scatter(train_data['Sex'].loc[train_data['Survived']==1],train_data['Age'].loc[train_data['Survived']==1],c='green',marker='x',alpha=.8)

plt.scatter(train_data['Sex'].loc[train_data['Survived']==0],train_data['Age'].loc[train_data['Survived']==0],c='red',marker='x',alpha=.2)

plt.xlabel('Sex=>0:male,1:female')

plt.ylabel('Age')

plt.show()
temp = pd.crosstab(train_data['Sex'],train_data['Survived'])

temp.plot(kind='bar',stacked=True,color=['red','blue'],grid=False)

plt.show()

print(temp)
frames = [train_data_X,test_data]

combined = pd.concat(frames,ignore_index=True)

combined.tail()
combined.head()
combined['Female'] = combined['Sex']

combined = combined.drop(['Sex'],axis=1)

combined.head()
combined.isnull().sum()
combined_null = combined.loc[combined['Age'].isnull()]

combined_notnull = combined.loc[combined['Age'].notnull()]
print(combined_null.head())

print(combined_notnull.head())
#temp = combined_notnull.pivot_table(values=['Age'],index=['Female','Pclass','Embarked','Parch','SibSp'],aggfunc=np.mean)

plt.hist(combined_notnull.Age,bins=50)

plt.show()
def binning(col,cut_pts,labels=None

            ):

    minval=col.min()

    maxval=col.max()

    break_pts = [minval]+cut_pts+[maxval]

    if not labels:

        labels = range(len(cut_pts)+1)

    col_bins=pd.cut(col,bins=break_pts,labels=labels,include_lowest=True)

    return col_bins



cut_pts =[15,45]

labels=['young','mid','old']

combined_notnull['Age_bins'] = binning(combined_notnull['Age'],cut_pts,labels)
combined_notnull.head()
train_data.head()
train_data['Age'] = train_data.Age.fillna(0)

train_data.isnull().sum()
cut_pts=[0.1,5,70]

labels=['null','kids','adults','seniors']

train_data['Age_bins'] = binning(train_data['Age'],cut_pts,labels)
t = pd.crosstab(train_data['Age_bins'],train_data['Survived'])

t['prob_of_survival'] = (t[1].values)/(t[1].values+t[0].values)

print(t)
print(combined.index)

combined.isnull().sum()
combined['isnull'] = 0

combined['isnull'].ix[combined['Age'].isnull()] = 1

combined['iskid'] = 0

combined['iskid'].loc[combined.Age<=5] = 1
combined['iskid'].loc[combined['iskid']==1].shape
combined = combined.drop(['Age'],axis=1)

combined.shape
train_data_X = combined.iloc[:891,:]

test_data = (combined.iloc[891:,:]).reset_index()

print(train_data_X.index)

print(test_data.index)
combined.head()
t = pd.crosstab(train_data_X['Embarked'],train_data_y)

t['prob_of_survival'] = (t[1].values)/(t[1].values+t[0].values)

t
plt.boxplot(train_data_X['Fare'])

plt.show()
from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import scale

#train_data_X = scale(train_data_X)

X_train,X_test,y_train,y_test=train_test_split(train_data_X,train_data_y,test_size=0.3,random_state=25)

svc_model = svm.SVC(C=100,gamma=.01,kernel='rbf')

svc_model.fit(train_data_X,train_data_y)

svc_model.score(X_test,y_test)
test_data['isSingle'] = 0

single = test_data.loc[test_data['Parch']==0].loc[test_data['SibSp']==0].index

test_data['isSingle'].loc[single]=1
plt.figure(figsize=(16,8))

plt.subplot(122)

plt.hist(train_data_X['Fare'],bins=50)

plt.xticks(np.arange(0,500,25),fontsize=8)

plt.subplot(121)

plt.hist(train_data_X['Fare'].loc[train_data_y==1],color='green',alpha=0.6,bins=50)

plt.hist(train_data_X['Fare'].loc[train_data_y==0],color='red',alpha=0.4,bins=50)

plt.xticks(np.arange(0,500,25),fontsize=8)

plt.show()
cut_pts=[5,20,80]

labels=[0,1,2,3]

train_data_X['Fare_bins'] = binning(train_data_X['Fare'],cut_pts,labels)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)

#train_data_scaled = scale(train_data_X)

X_train,X_test,y_train,y_test=train_test_split(train_data_X,train_data_y,test_size=0.3,random_state=25)

clf = clf.fit(train_data_X,train_data_y)

#test_data_scaled = scale(test_data)

y_pred = clf.predict(test_data)

sc = clf.score(train_data_X,train_data_y)

sc
