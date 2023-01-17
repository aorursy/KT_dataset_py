import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
data = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
data.shape, test.shape
data.columns
data.head()
test.describe()
data.dtypes

Survived_m = data.Survived[data['Sex']=='male'].value_counts()
Survived_f = data.Survived[data["Sex"]=='female'].value_counts()

df = pd.DataFrame({'male': Survived_m, 'female':Survived_f})
df.plot(kind= 'bar', stacked= True)
plt.xlabel('survived')
plt.ylabel("number")
plt.title('survival number against sex')
plt.show()
data['Age'].hist()
plt.xlabel("age")
plt.ylabel("number")
plt.title("age distribution")
plt.show()


data['Age'][data.Survived ==0].hist()
plt.xlabel("age")
plt.ylabel("number")
plt.title("unsurvived age distribution")
plt.show()

data['Age'][data.Survived ==1].hist()
plt.xlabel("age")
plt.ylabel("number")
plt.title("survived age distribution")
plt.show()
data['Fare'].hist()
plt.xlabel("Fare")
plt.ylabel("number")
plt.title("fare distribution")
plt.show()

data['Fare'][data.Survived ==0].hist()
plt.xlabel("Fare")
plt.ylabel("number")
plt.title("unsurvived fare distribution")
plt.show()

data['Fare'][data.Survived ==1].hist()
plt.xlabel("Fare")
plt.ylabel("number")
plt.title("survived fare distribution")
plt.show()
data['Pclass'].hist()
plt.xlabel("class")
plt.ylabel("number")
plt.title("class distribution")
plt.show()

data['Pclass'][data.Survived ==0].hist()
plt.xlabel("class")
plt.ylabel("number")
plt.title("unsurvived class distribution")
plt.show()

data['Pclass'][data.Survived ==1].hist()
plt.xlabel("class")
plt.ylabel("number")
plt.title("survived class distribution")
plt.show()
Embaked_Q = data.Survived[data['Embarked'] =='Q'].value_counts()
Embaked_S = data.Survived[data['Embarked'] =='S'].value_counts()
Embaked_C = data.Survived[data['Embarked'] =='C'].value_counts()

df1= pd.DataFrame({'S':Embaked_S, 'Q':Embaked_Q,'C':Embaked_C})
df1.plot(kind ='bar',stacked=True)
plt.xlabel('Survived')
plt.ylabel("number")
plt.show()
dataSet = data.loc[:, ['Pclass','Sex','Age','Fare','Embarked']]
testSet = test.loc[:, ['Pclass','Sex','Age','Fare','Embarked']]

dataSet.shape, testSet.shape
print(dataSet)
print(testSet)
labels = data.Survived
def fill_na(data):
    data_copy = data.copy(deep=True)
    data_copy['Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy['Age'] = data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy['Sex'] = data_copy['Sex'].fillna('female')
    data_copy['Embarked'] = data_copy['Embarked'].fillna('S')
    data_copy['Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    return data_copy

data_noNAN = fill_na(dataSet)
test_noNAN = fill_na(testSet)

data_noNAN.isnull().any(),test_noNAN.isnull().any()
def transfer_sex(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Sex'] =='male','Sex'] =1
    data_copy.loc[data_copy['Sex'] =='female','Sex'] =0

    return data_copy
data_afterSex = transfer_sex(data_noNAN)
test_afterSex = transfer_sex(test_noNAN)

def transfer_embarked(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Embarked'] =='Q','Embarked']=0
    data_copy.loc[data_copy['Embarked'] =='C','Embarked']=1
    data_copy.loc[data_copy['Embarked'] =='S','Embarked']=2
    return data_copy
data_afterEmbarked = transfer_embarked(data_afterSex)
test_afterEmbarked = transfer_embarked(test_afterSex)

train_now = data_afterEmbarked
test_now = test_afterEmbarked
test_now
from sklearn.model_selection import train_test_split
X_train,X_vali,y_train, y_vali = train_test_split(train_now,
                                                 labels,
                                                 test_size= 0.2,
                                                  random_state=0)
X_train.shape,X_vali.shape,y_train.shape, y_vali.shape
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

k_range = range(1,50)
k_scores = []
for k in k_range:
    print("k= " + str(k) + " started" )
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(X_train,y_train)
    predictions = clf.predict(X_vali)
    score = accuracy_score(y_vali,predictions)
    print("accuracy = " +str(score))
    k_scores.append(score)
    
# 预测
plt.plot(k_range,k_scores)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()
# 检测模型precision， recall 等各项指标
np.argsort(k_scores)
clf1= KNeighborsClassifier(n_neighbors = 33)
clf1.fit(train_now,labels)
result = clf1.predict(test_now)
result
# 预测
df = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': result})
df.to_csv("submission.csv", header =True, index = False)
df.Survived.value_counts()

