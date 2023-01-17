import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
train.columns
print(train.shape)

print(test.shape)
train.head()
train.describe()
def check_missing(data):

    for i in data.columns:

        outcome = data[str(i)].isnull().values.any()

        if outcome==False:

            continue

        print(i)

print('train:')       

check_missing(train)

print('\ntest:')

check_missing(test)
#Sex没有missing values

male = train['Survived'][train['Sex'] == 'male'].value_counts()

female = train['Survived'][train['Sex'] == 'female'].value_counts()



df=pd.DataFrame({'male':male, 'female':female})

df.plot(kind='bar', stacked=True)

plt.title("survived by sex")

plt.xlabel("survived") 

plt.ylabel("count")

plt.show()
train['Age'].hist(label='total')  

plt.ylabel("Number") 

plt.xlabel("Age") 

plt.title('Age distribution')



train[train.Survived==0]['Age'].hist(label='victims')  

plt.ylabel("Number") 

plt.xlabel("Age") 



train[train.Survived==1]['Age'].hist(label='survivors')  

plt.ylabel("Number") 

plt.xlabel("Age") 

plt.legend()
train['Fare'].hist()  

plt.ylabel("Number") 

plt.xlabel("Fare") 

plt.title('Fare distribution')

plt.show() 



train[train.Survived==0]['Fare'].hist()  

plt.ylabel("Number") 

plt.xlabel("Fare") 

plt.title('Fare distribution of victims')

plt.show()



train[train.Survived==1]['Fare'].hist()  

plt.ylabel("Number") 

plt.xlabel("Fare") 

plt.title('Fare distribution of survivors')

plt.show()
train['Pclass'].hist()  

plt.xticks(np.arange(1, 4, step=1))

plt.show()  



p1 = train.Survived[train['Pclass'] == 1].value_counts()

p2 = train.Survived[train['Pclass'] == 2].value_counts()

p3 = train.Survived[train['Pclass'] == 3].value_counts()



df=pd.DataFrame({'p1':p1, 'p2':p2, 'p3':p3})

print(df)

df.plot(kind='bar', stacked=True)

plt.title("survived by pclass")

plt.xlabel("pclass") 

plt.ylabel("count")

plt.show()
train['Embarked'].unique()
S = train['Survived'][train['Embarked']=='S'].value_counts()

C = train['Survived'][train['Embarked']=='C'].value_counts()

Q = train['Survived'][train['Embarked']=='Q'].value_counts()



embarked = pd.DataFrame({'S':S,"C":C,'Q':Q})

print(embarked)

embarked.plot(kind='bar',stacked=True)

plt.xlabel('port')

plt.ylabel('count')

plt.show()
label = train['Survived']

trainset = train[['Pclass','Sex','Age','Fare','Embarked']]

testset = test[['Pclass','Sex','Age','Fare','Embarked']]

print(label.shape)

print(trainset.shape)

print(testset.shape)
# 已知train中age和embarked中有空数据，test中fare有空数据

def filling_nan(data):

    df = data.copy()

    df['Age'].fillna(df['Age'].median(),inplace=True)

    df['Embarked'].fillna('S',inplace=True)

    df['Fare'].fillna(df['Fare'].median(),inplace=True)

    return df



train_non_na = filling_nan(trainset)

test_non_na = filling_nan(testset)
check_missing(train_non_na)
check_missing(test_non_na)
def encode(data):

    df = data.copy()

    df['Sex'].replace({'male':1,'female':0},inplace=True)

    df['Embarked'].replace({'S':0,'C':1, 'Q':2},inplace=True)

    return df



train_clean = encode(train_non_na)

test_clean = encode(test_non_na)
train_clean
from sklearn.model_selection import train_test_split



train_data, val_data, train_labels, val_labels = train_test_split(train_clean,label,random_state=0,test_size=0.2)

print(train_data.shape, val_data.shape, train_labels.shape, val_labels.shape)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



k_range=range(1,51)

k_score=[]



for k in k_range:

    clf = KNeighborsClassifier(n_neighbors = k)

    clf.fit(train_data,train_labels)

    print('k=',k)

    pred = clf.predict(val_data)

    score = accuracy_score(val_labels,pred)

    k_score.append(score)

    print('score:',score)
plt.plot(k_range,k_score)

plt.xlabel('k')

plt.ylabel('accuracy on validation set')

plt.title('KNN on predicting survivors in Titanic')

plt.show()
# 预测

k_and_scores = list(zip(k_range,k_score))

sorted(k_and_scores, key = lambda x:x[1], reverse=True)
#预测

model=KNeighborsClassifier(n_neighbors=33)

model.fit(train_data,train_labels)

knn_result=model.predict(val_data)
print(knn_result)
# 检测模型precision， recall 等各项指标

from sklearn.metrics import precision_score

precision = precision_score(val_labels, knn_result)

precision
from sklearn.metrics import recall_score

recall = recall_score(val_labels, knn_result)

recall
# 预测

test_result = model.predict(test_clean)
print(test_result)
outcome = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':test_result})

outcome
outcome.to_csv('submission.csv',header=True, index=False)