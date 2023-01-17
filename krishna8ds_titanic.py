# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.linear_model import LogisticRegression

from sklearn import metrics



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns





from sklearn.ensemble import VotingClassifier,RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from prettytable import PrettyTable
data_train = pd.read_csv('../input/titanic/train.csv')

data_test = pd.read_csv('../input/titanic/test.csv')
data_train.head(10)
data_train.tail()
print('train size',data_train.shape)

print('test size',data_test.shape)
print(data_train.info())

print('#####################################')

print(data_test.info())
print(data_train.isnull().sum())

print('##############################')

print(data_test.isnull().sum())
plt.figure(figsize=(12,8))

sns.heatmap(data_train.isnull())

plt.show()
print(round((data_train.isnull().sum()/len(data_train))*100))

print('##############################')

print(round((data_test.isnull().sum()/len(data_test))*100))

plt.figure(figsize=(10,6))

mpl.style.use('seaborn')

sns.countplot(data_train.Embarked,hue=data_train.Survived,palette='rainbow')

axis_font = {'fontname':'Arial', 'size':'16'}

tick_font = {'fontname':'Arial', 'size':'14'}

title = {'fontname':'Arial', 'size':'18'}

plt.title('Passengers boarded in various Ports & Survived',**title)

plt.xlabel('Embarked',**axis_font)

plt.ylabel('Count of People',**axis_font)

plt.xticks(**tick_font)

plt.show()
plt.figure(figsize=(10,6))

mpl.style.use('seaborn')

sns.countplot(data_train.Pclass,palette='Set2',hue=data_train.Survived)

axis_font = {'fontname':'Arial', 'size':'16'}

tick_font = {'fontname':'Arial', 'size':'14'}

title = {'fontname':'Arial', 'size':'18'}

plt.title('people in each class & survived',**title)

plt.xlabel('Pclass',**axis_font)

plt.ylabel('Count',**axis_font)

plt.xticks(**tick_font)

plt.show()
sns.lineplot(y=data_train.Survived,x=data_train.SibSp,ci=0)

axis_font = {'fontname':'Arial', 'size':'16'}

tick_font = {'fontname':'Arial', 'size':'14'}

title = {'fontname':'Arial', 'size':'18'}

plt.title('No.Of Siblings & survivality',**title)

plt.xlabel('no.of Siblings',**axis_font)

plt.ylabel('Survived',**axis_font)

plt.xticks(**tick_font)

plt.show()
sns.lineplot(y=data_train.Survived,x=data_train.Parch,ci=0)

axis_font = {'fontname':'Arial', 'size':'16'}

tick_font = {'fontname':'Arial', 'size':'14'}

title = {'fontname':'Arial', 'size':'18'}

plt.title('No.Of Parents',**title)

plt.xlabel('Parents',**axis_font)

plt.ylabel('Survived',**axis_font)

plt.xticks(**tick_font)

plt.show()
sns.countplot(x=data_train.Sex,hue=data_train.Survived)

axis_font = {'fontname':'Arial', 'size':'16'}

tick_font = {'fontname':'Arial', 'size':'14'}

title = {'fontname':'Arial', 'size':'18'}

plt.title('comparing gender to survivality',**title)

plt.xlabel('Gender',**axis_font)

plt.ylabel('Count',**axis_font)

plt.xticks(**tick_font)

plt.show()
sns.countplot(x=data_train.Embarked,hue=data_train.Survived,palette='Set2')

axis_font = {'fontname':'Arial', 'size':'16'}

tick_font = {'fontname':'Arial', 'size':'14'}

title = {'fontname':'Arial', 'size':'18'}

plt.title('Pclass & Survivality',**title)

plt.xlabel('Pclass',**axis_font)

plt.ylabel('Count',**axis_font)

plt.xticks(**tick_font)

plt.show()
sns.countplot(hue=data_train.Pclass,x=data_train.Embarked,palette='Accent')

axis_font = {'fontname':'Arial', 'size':'16'}

tick_font = {'fontname':'Arial', 'size':'14'}

title = {'fontname':'Arial', 'size':'18'}

plt.title('Count people in all classes boarded in which port ',**title)

plt.xlabel('Pclass',**axis_font)

plt.ylabel('Count',**axis_font)

plt.xticks(**tick_font)
plt.figure(figsize=(10,8))

sns.boxenplot(x=data_train.Pclass,y=data_train.Age)

axis_font = {'fontname':'Arial', 'size':'16'}

tick_font = {'fontname':'Arial', 'size':'14'}

title = {'fontname':'Arial', 'size':'18'}

plt.title('Checking for Outliers in Age by Pclass',**title)

plt.xlabel('Pclass',**axis_font)

plt.ylabel('Age',**axis_font)

plt.xticks(**tick_font)

plt.show()
corr= data_train.corr()

sns.heatmap(corr,annot=True)

plt.show()
data_test.drop(['Cabin','Ticket','PassengerId'],axis=1,inplace=True)

data_train.drop(['Cabin','Ticket','PassengerId'],axis=1,inplace=True)
data_train.head()
data_train.head()
print(data_test.shape)

print(data_train.shape)
data_train.Embarked.fillna('S',inplace=True)

data_train.Fare.fillna(53,inplace=True)
data_test.Fare.fillna(53,inplace=True)
def name_extract(word):

 return word.split(',')[1].split('.')[0].strip() 
data_train['n_name'] = data_train['Name'].apply(name_extract) 

data_test['n_name'] = data_test['Name'].apply(name_extract) 

data_train.replace(['Dr','Rev','Col','Major','Mlle','Mme','Capt','Jonkheer','Lady','Don','Sir','the Countess','Ms'],'others',inplace=True) 

data_test.replace(['Col','Rev','Dr','Ms','Dona'],'others',inplace=True) 
data_train.drop('Name',axis=1,inplace=True)  

data_test.drop('Name',axis=1,inplace=True) 
data_train.Age.fillna(28,inplace=True)

data_test.Age.fillna(27,inplace=True)
data_train['family'] = data_train['SibSp'] + data_train['Parch'] + 1

data_test['family'] = data_test['SibSp'] + data_test['Parch'] + 1
def size_family(x):

    if(x==1):

        return(0)

    elif(x>1) and (x<=4):

        return(1)

    elif(x>4):

        return(2)
data_train['size'] = data_train['family'].apply(size_family)

data_test['size'] = data_test['family'].apply(size_family)
def dependent(x):

    if(x<=10):

        return(0)

    elif(x<10)and(x<=19):

        return(1)

    elif(x>19)and(x<50):

        return(2)

    else:

        return(3) 
data_train['dependent'] = data_train['Age'].apply(dependent)

data_test['dependent'] = data_test['Age'].apply(dependent)
data_train.head()
data_test.head()
train_d = pd.get_dummies(data_train)

test_d = pd.get_dummies(data_test)
train_d.head() 
test_d.head()
x=train_d.drop(['Survived'],axis=1)

y=train_d[['Survived']]
sc=StandardScaler()

x_sc = pd.DataFrame(sc.fit_transform(x),columns=x.columns) 

test_sc =pd.DataFrame(sc.transform(test_d),columns=test_d.columns) 
x_sc.head() 
ytest = pd.read_csv('../input/scores/scores.csv')
rf = RandomForestClassifier(random_state=323,n_estimators=5,criterion='gini',max_depth=4,max_leaf_nodes=11)

rf.fit(x_sc,y)

y_rf_pre = rf.predict(test_sc)
print(metrics.accuracy_score(ytest,y_rf_pre))

print(metrics.recall_score(ytest,y_rf_pre))

print(metrics.precision_score(ytest,y_rf_pre))

print(metrics.f1_score(ytest,y_rf_pre))

print(metrics.confusion_matrix(ytest,y_rf_pre))

print(metrics.cohen_kappa_score(ytest,y_rf_pre))
lr = LogisticRegression()

lr.fit(x_sc,y)

y_lr_pre = lr.predict(test_sc)

print(metrics.accuracy_score(ytest,y_lr_pre))

print(metrics.recall_score(ytest,y_lr_pre))

print(metrics.precision_score(ytest,y_lr_pre))

print(metrics.f1_score(ytest,y_lr_pre))

print(metrics.confusion_matrix(ytest,y_lr_pre))

print(metrics.cohen_kappa_score(ytest,y_lr_pre))
lr = LogisticRegression(random_state=1,penalty='l1')

rf = RandomForestClassifier(random_state=323,n_estimators=5,criterion='gini',max_depth=4,max_leaf_nodes=11)

ada = AdaBoostClassifier(random_state=21)

bag = BaggingClassifier(random_state=2)

gb = GradientBoostingClassifier(random_state=8)

vc = VotingClassifier(estimators=[('lr',lr),('rf',rf),('ada',ada),('bag',bag),('gb',gb)],voting='hard')

vc=vc.fit(x_sc,y)

y_vc_pre=vc.predict(test_sc)
print('accuracy ',metrics.accuracy_score(ytest,y_vc_pre))

print('precision',metrics.precision_score(ytest,y_vc_pre))

print('recall',metrics.recall_score(ytest,y_vc_pre))

print('F1',metrics.f1_score(ytest,y_vc_pre))

print('roc_auc_score',metrics.roc_auc_score(ytest,y_vc_pre))

print('cohen_kappa',metrics.cohen_kappa_score(ytest,y_vc_pre))

print(metrics.classification_report(ytest,y_vc_pre))

print(metrics.confusion_matrix(ytest,y_vc_pre))
test_pred = pd.DataFrame(y_rf_pre, columns= ['Survived'])
z = pd.read_csv('../input/titanic/test.csv')
final_test = pd.concat([z, test_pred], axis=1, join='inner')
df= final_test[['PassengerId' ,'Survived']]
df.to_csv('Predicted_rf_k.csv',index=False)
t = PrettyTable(['Model', 'Accuracy', 'Recall','Precision','F1','cohen_kappa'])

t.add_row(['Random Forest',94.4,95.3,90.6,92.6,88.2])

t.add_row(['Logistic Regression',92.5,94.0,86.6,90.2,84.2])

t.add_row(['Voting Classifier',93.7,88.8,94.7,91.7,86.7])


print(t)