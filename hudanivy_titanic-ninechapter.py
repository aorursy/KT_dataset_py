import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
dataset=pd.read_csv('../input/train.csv')
testset=pd.read_csv('../input/test.csv')

dataset.head(20)
dataset.corr()['Survived']
survived_male=dataset.loc[dataset['Sex']=='male','Survived'].value_counts()
survived_female=dataset.loc[dataset['Sex']=='female','Survived'].value_counts()
#can also be written as 
#survived_male=dataset.Survived[dataset.Sex=='male'].value_counts()
#survived_female=dataset.Survived[dataset.Sex=='female'].value_counts()
df=pd.DataFrame({'survived_male':survived_male,'survived_female':survived_female})
df.plot(kind='bar',stacked=True)
plt.title('survive rate by sex')
plt.xlabel('survived or not')
plt.ylabel('count')
plt.show()
dataset['Age'].plot(kind='hist')
plt.title('Age of all')
plt.xlabel('Age')
plt.ylabel('count')
plt.show()

dataset.Age[dataset.Survived ==1].plot(kind='hist')
plt.title('Age of Survived')
plt.xlabel('Age')
plt.ylabel('count')
plt.show()

dataset.Age[dataset.Survived ==0].plot(kind='hist')
plt.title('Age of Not Survived')
plt.xlabel('Age')
plt.ylabel('count')
plt.show()
dataset['Fare'].plot(kind='hist')
plt.title('Fare of all')
plt.xlabel('Fare')
plt.ylabel('count')
plt.show()

dataset.Fare[dataset.Survived ==1].plot(kind='hist')
plt.title('Fare of Survived')
plt.xlabel('Fare')
plt.ylabel('count')
plt.show()

dataset.Fare[dataset.Survived ==0].plot(kind='hist')
plt.title('Fare of Not Survived')
plt.xlabel('Fare')
plt.ylabel('count')
plt.show()
survived_P1=dataset.loc[dataset['Pclass']==1,'Survived'].value_counts()
survived_P2=dataset.loc[dataset['Pclass']==2,'Survived'].value_counts()
survived_P3=dataset.loc[dataset['Pclass']==3,'Survived'].value_counts()

df=pd.DataFrame({'P1':survived_P1,'P2':survived_P2,'P3':survived_P3})
df.plot(kind='bar',stacked=True)
plt.title('survive rate by Pclass')
plt.xlabel('survived or not')
plt.ylabel('count')
plt.show()
S=dataset.loc[dataset['Embarked']=='S','Survived'].value_counts()
Q=dataset.loc[dataset['Embarked']=='Q','Survived'].value_counts()
C=dataset.loc[dataset['Embarked']=='C','Survived'].value_counts()

df=pd.DataFrame({'S':S,'Q':Q,'C':C})
df.plot(kind='bar',stacked=True)
plt.title('survive rate by Embarked place')
plt.xlabel('survived or not')
plt.ylabel('count')
plt.show()
label=dataset.loc[:,'Survived']
label.head()
dataset1=dataset.loc[:,['Pclass','Sex','Age','Fare','Embarked']]
testset1=testset.loc[:,['Pclass','Sex','Age','Fare','Embarked']]
print(dataset1.head())
print(testset1.head())


def fill_Nan(data):
    data_copy=data.copy(deep=True)
    data_copy['Sex']=data_copy['Sex'].fillna('male')
    data_copy.loc[data_copy['Sex']=='male','Age']=data_copy.loc[data_copy['Sex']=='male','Age'].fillna(data_copy.loc[data_copy['Sex']=='male','Age'].median())
    data_copy.loc[data_copy['Sex']=='female','Age']=data_copy.loc[data_copy['Sex']=='female','Age'].fillna(data_copy.loc[data_copy['Sex']=='female','Age'].median())
    data_copy.loc[:,'Embarked']=data_copy['Embarked'].fillna('S')
    data_copy.loc[:,'Fare']=data_copy['Fare'].fillna(data_copy['Fare'].median())
   # data_copy['Parch']=data_copy['Parch'].fillna(data_copy['Parch'].median())
    return data_copy
dataset1 = fill_Nan(dataset1)
testset1 = fill_Nan(testset1)
#print (testset1.isnull().values.any())

#print(dataset1.head())
dataset1.loc[dataset1['Sex']=='male','Sex'] = 0
dataset1.loc[dataset1['Sex']=='female','Sex'] = 1

testset1.loc[testset1['Sex']=='male','Sex'] = 0
testset1.loc[testset1['Sex']=='female','Sex'] = 1

#print(dataset1.head())
dataset1.loc[dataset1['Embarked']=='S','Embarked'] = 0
dataset1.loc[dataset1['Embarked']=='Q','Embarked'] = 1
dataset1.loc[dataset1['Embarked']=='C','Embarked'] = 2
#print (dataset1.head())
testset1.loc[testset1['Embarked']=='S','Embarked'] = 0
testset1.loc[testset1['Embarked']=='Q','Embarked'] = 1
testset1.loc[testset1['Embarked']=='C','Embarked'] = 2

from sklearn.model_selection import train_test_split
train_set,test_set,train_label,test_label=train_test_split(dataset1,label,random_state=1,test_size=0.2)
print('train_set shape:',train_set.shape,'test_set shape:',test_set.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
accuracy=[]
for K in range(1,51):
    clf=KNeighborsClassifier(n_neighbors=K)
    clf.fit(train_set,train_label)
    prediction=clf.predict(test_set)
    tmp_accuracy=accuracy_score(test_label,prediction)
    #print ("K=",K,"accuracy=",tmp_accuracy)
    accuracy.append(tmp_accuracy)

# 预测
print (np.array(accuracy).argsort())
    
# 检测模型precision， recall 等各项指标

# cross validation 找到最好的k值
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=0,shuffle=True)
#X_train,Y_train,X_test,Y_test=kf.split(dataset1,label)
i=1
for train_index,test_index in kf.split(dataset1):
    #print ("train_index:",train_index[:5],"test_index:",test_index[:5])
    train_set,train_label,test_set,test_label=dataset1.loc[train_index],label.loc[train_index],dataset1.loc[test_index],label.loc[test_index]
    accuracy=[]  
    for K in range(1,51):
        clf=KNeighborsClassifier(n_neighbors=K)
        clf.fit(train_set,train_label)
        prediction=clf.predict(test_set)
        tmp_accuracy=accuracy_score(test_label,prediction)
        accuracy.append(tmp_accuracy)
    plt.plot(range(1,51),accuracy)
    plt.show()
    print ("fold ",i,"best K=", np.array(accuracy).argsort()[-1]+1)
    i +=1
    
    
from sklearn.model_selection import cross_val_score
score_list=[]
for K in range(1,51):
    clf=KNeighborsClassifier(n_neighbors=K)
    score_list.append(cross_val_score(clf, dataset1, label,cv=5).mean())
print("best K=",np.array(score_list).argsort()[-1]+1,"best average accuracy of this K=",max(score_list))

# 预测
clf=KNeighborsClassifier(n_neighbors=3)
clf.fit(dataset1,label)
#print(testset.shape)
prediction=clf.predict(testset1)
#testset.head()
#prediction[:20]
df=pd.DataFrame({"PassengerId":testset.PassengerId,"Survived":prediction})
df.to_csv("dandan_titanic_submission.csv",header=True)
