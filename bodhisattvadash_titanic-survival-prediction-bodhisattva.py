# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_set = pd.read_csv('../input/train.csv')

print(train_set.shape)

train_set.describe()
test_set = pd.read_csv('../input/test.csv')

print(train_set.shape)

test_set.describe()
train_set.info()

test_set.info()
train_set.isna().sum()

test_set.isna().sum()
train_cor = train_set.corr()

plt.figure(figsize = (8,8))

sns.heatmap(train_cor, annot = True)

plt.title('Correlation of features in train dataset')

plt.show()
train_set[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values('Survived', ascending=False)                            
train_set[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values('Survived', ascending=False)  
train_set = train_set.drop(['Cabin','Ticket'], axis = 1)
test_set = test_set.drop(['Cabin','Ticket'], axis = 1)
train_set['FamilySize1']=train_set['SibSp']+train_set['Parch']+1

test_set['FamilySize1']=test_set['SibSp']+test_set['Parch']+1
train_set['FamilySize1'].value_counts()

train_set[['FamilySize1','Survived']].groupby(['FamilySize1'], as_index=False).mean().sort_values('Survived', ascending=False) 
train_set['FamilySizeGroup']=pd.cut(train_set['FamilySize1'],4)

test_set['FamilySizeGroup']=pd.cut(test_set['FamilySize1'],4)

train_set[['FamilySizeGroup','Survived']].groupby(['FamilySizeGroup']).mean().sort_values('Survived',ascending=False) 
combined_set_new = [train_set, test_set]

for fg in combined_set_new:

    fg['FSC'] = 0;

    fg.loc[fg['FamilySize1']>3.5,'FSC']=3

    fg.loc[(fg['FamilySize1']>0.99) & (fg['FamilySize1']<=3.5),'FSC']=2

    fg.loc[(fg['FamilySize1']>6) & (fg['FamilySize1']<=8.5),'FSC']=1

    fg.loc[(fg['FamilySize1']>8.5) & (fg['FamilySize1']<=11),'FSC']=0   
combined_set=[train_set, test_set]

for per in combined_set:

    per['IsAlone'] = 0;

    per.loc[per['FamilySize1']==1,'IsAlone']=1
train_set[['IsAlone','Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values('Survived', ascending=False)  
train_set['Embarked'].value_counts()

train_set['Embarked']=train_set['Embarked'].fillna('S')
train_set[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean().sort_values('Survived', ascending=False)   
train_set['EmbarkedMap']=train_set['Embarked'].map({'S':0,'Q':1,'C':2})

test_set['EmbarkedMap']=test_set['Embarked'].map({'S':0,'Q':1,'C':2})
g=sns.FacetGrid(train_set,col='Survived')

g.map(plt.hist,'Fare')


train_set['FareGroup']=pd.cut(train_set['Fare'],4)

train_set[['FareGroup','Survived']].groupby(['FareGroup']).mean().sort_values('Survived',ascending=False)
for fg in combined_set:

    fg['FGC'] = 0;

    fg.loc[fg['Fare']>384.247,'FGC']=3

    fg.loc[(fg['Fare']>128.082) & (fg['Fare']<=256.165),'FGC']=2

    fg.loc[(fg['Fare']>256.165) & (fg['Fare']<=384.247),'FGC']=1

    fg.loc[fg['Fare']<=128.082,'FGC']=0    
train_set['Title1']=train_set['Name'].str.extract('([A-Za-z]+)\.', expand=False)

test_set['Title1']=test_set['Name'].str.extract('([A-Za-z]+)\.', expand=False)

train_set[['Title1', 'Survived']].groupby(['Title1'], as_index=False).mean().sort_values('Survived',ascending=False)
combined_set_new = [train_set, test_set]

for cn in combined_set_new:

    cn['Title1'] = cn['Title1'].replace(['Sir','Countess','Master','Col','Major','Dr','Jonkheer','Don','Rev','Capt'],'Others')

    cn['Title1'] = cn['Title1'].replace(['Lady','Mme'],'Mrs')

    cn['Title1'] = cn['Title1'].replace(['Miss','Mlle', 'Ms'],'Miss')

train_set[['Title1','Survived']].groupby(['Title1'], as_index = False).mean().sort_values(['Survived'], ascending = False)

train_set['TitleMap']=train_set['Title1'].map({'Mr':1,'Others':2,'Miss':3,'Mrs':4})

test_set['TitleMap']=test_set['Title1'].map({'Mr':1,'Others':2,'Miss':3,'Mrs':4})

test_set['TitleMap']=test_set['TitleMap'].fillna(0)
plt.figure(figsize=(8,8))

sns.kdeplot(train_set.loc[train_set.Survived == 0, 'Age'], label = 'Survived == 0')

sns.kdeplot(train_set.loc[train_set.Survived == 1, 'Age'], label = 'Survived == 1')

plt.xlabel('Age')

plt.ylabel('Survived Density')

plt.show()
train_set['Age']=train_set['Age'].fillna(train_set['Age'].mean())

test_set['Age']=test_set['Age'].fillna(train_set['Age'].mean())
plt.figure(figsize=(8,8))

sns.kdeplot(train_set.loc[train_set.Survived == 0, 'Fare'], label = 'Survived == 0')

sns.kdeplot(train_set.loc[train_set.Survived == 1, 'Fare'], label = 'Survived == 1')

plt.xlabel('Fare')

plt.ylabel('Survived Density')

plt.show()
train_set['Fare']=train_set['Fare'].fillna(train_set['Fare'].median())

test_set['Fare']=test_set['Fare'].fillna(train_set['Fare'].median())
train_set['AgeGroup']=pd.cut(train_set['Age'],4)

test_set['AgeGroup']=pd.cut(test_set['Age'],4)

train_set[['AgeGroup','Survived']].groupby(['AgeGroup']).mean().sort_values('Survived',ascending=False)
combined_set_new_1 = [train_set, test_set]
for fg in combined_set_new_1:

    fg['AGC'] = 0;

    fg.loc[(fg['Age']>0.34) & (fg['Age']<=20.315),'AGC']=3

    fg.loc[(fg['Age']>40.21) & (fg['Age']<=60.105),'AGC']=2

    fg.loc[(fg['Age']>20.315) & (fg['Age']<=40.21),'AGC']=1

    fg.loc[(fg['Age']>60.105) & (fg['Age']<=80),'AGC']=0   

    fg['SC']=0

    fg.loc[fg['Sex']=='male','SC']=0

    fg.loc[fg['Sex']=='female','SC']=1
train_set_class = train_set['Survived']

train_set_new = train_set.drop(['PassengerId','Survived','Name','SibSp','Parch', \

                               'Embarked','FamilySizeGroup','FareGroup','Title1','AgeGroup','Sex'], axis = 1).copy()
test_set_new = test_set.drop(['PassengerId','Name','SibSp','Parch', \

                               'Embarked','FamilySizeGroup','Title1','AgeGroup','Sex'], axis = 1).copy()
train_set_new.shape, train_set_class.shape, test_set_new.shape

tr_set, te_set, tr_cl, te_cl = train_test_split(train_set_new, train_set_class, test_size = 0.3, random_state = 42)

logregclassi = LogisticRegression()

logregclassi.fit(tr_set, tr_cl)

accuracy_obt_tr = round(logregclassi.score(tr_set, tr_cl)*100,2)

accuracy_obt_te = round(logregclassi.score(te_set, te_cl)*100,2)

te_Predict = logregclassi.predict(te_set)

print('The training and testing accuracy obtained is {}, and {}, respectively.'.format(accuracy_obt_tr,accuracy_obt_te))
print(confusion_matrix(te_cl,te_Predict))

print(classification_report(te_cl,te_Predict))
Final_Predict = logregclassi.predict(test_set_new)

sub_logc = pd.DataFrame()

sub_logc['PassengerId'] = test_set['PassengerId']

sub_logc['Survived'] = Final_Predict

sub_logc.to_csv('LogisticRegression.csv',index=False)
from sklearn.svm import SVC
svc = SVC()

svc.fit(tr_set, tr_cl)

accuracy_obt_tr_svc = round(svc.score(tr_set, tr_cl)*100,2)

accuracy_obt_te_svc = round(svc.score(te_set, te_cl)*100,2)

te_Predict_svc = svc.predict(te_set)

print('The training and testing accuracy obtained is {}, and {}, respectively.'.format(accuracy_obt_tr_svc,accuracy_obt_te_svc))
Final_Predict = svc.predict(test_set_new)

sub_svcc = pd.DataFrame()

sub_svcc['PassengerId'] = test_set['PassengerId']

sub_svcc['Survived'] = Final_Predict

sub_svcc.to_csv('SVC.csv',index=False)
j=0

accuracy_obt_tr_knn = np.zeros((4))

accuracy_obt_te_knn = np.zeros((4))

for i in range(3,10,2):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(tr_set, tr_cl)

    accuracy_obt_tr_knn[j] = round(knn.score(tr_set, tr_cl)*100,2)

    accuracy_obt_te_knn[j] = round(knn.score(te_set, te_cl)*100,2)

    print('The training and testing accuracy obtained with n = {} is {}, and {}, respectively.'.format(i,accuracy_obt_tr_knn[j],accuracy_obt_te_knn[j]))

    j=j+1
Final_Predict = knn.predict(test_set_new)

sub_logk = pd.DataFrame()

sub_logk['PassengerId'] = test_set['PassengerId']

sub_logk['Survived'] = Final_Predict

sub_logk.to_csv('KNN.csv',index=False)
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(tr_set, tr_cl)

Y_pred = random_forest.predict(te_set)

acc_random_forest_tr = round(random_forest.score(tr_set, tr_cl) * 100, 2)

acc_random_forest_te = round(svc.score(te_set, te_cl)*100,2)

print('The training and testing accuracy obtained is {}, and {}, respectively.'.format(acc_random_forest_tr,acc_random_forest_te))
Final_Predict = knn.predict(test_set_new)

sub_logr = pd.DataFrame()

sub_logr['PassengerId'] = test_set['PassengerId']

sub_logr['Survived'] = Final_Predict

sub_logr.to_csv('Randomforet.csv',index=False)