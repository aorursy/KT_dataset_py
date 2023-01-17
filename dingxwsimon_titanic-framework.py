import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
dataset=pd.read_csv('../input/train.csv')
testset=pd.read_csv('../input/test.csv')
dataset.columns
dataset.head()
dataset.dtypes
dataset.describe()
print(dataset.isnull().values.any())
survived_m = dataset.Survived[dataset.Sex=='male'].value_counts()
survived_f = dataset.Survived[dataset.Sex=='female'].value_counts()
df = pd.DataFrame({'male': survived_m, 'female':survived_f})
df.plot(kind='bar', stacked=True)
plt.title('survived by sex')
plt.xlabel('suvival')
plt.ylabel('count')
plt.show()
dataset['Age'].hist()
plt.ylabel('Numbers')
plt.xlabel('Age')
plt.title('age distribtuion')
plt.show()

dataset[dataset.Survived == 0]['Age'].hist()
plt.ylabel('Numbers')
plt.xlabel('Age')
plt.title('age distribtuion of people did not survive')
plt.show()

dataset[dataset.Survived == 1]['Age'].hist()
plt.ylabel('Numbers')
plt.xlabel('Age')
plt.title('age distribtuion of people survive')
plt.show()
dataset['Fare'].hist(density=True)
plt.ylabel('Numbers')
plt.xlabel('Fare')
plt.title('Fare distribtuion')
plt.show()

dataset[dataset.Survived == 0]['Fare'].hist(density=True)
plt.ylabel('Numbers')
plt.xlabel('Fare')
plt.title('Fare distribtuion of people did not survive')
plt.show()

dataset[dataset.Survived == 1]['Fare'].hist(density=True)
plt.ylabel('Numbers')
plt.xlabel('Fare')
plt.title('Fare distribtuion of people survive')
plt.show()
dataset['Pclass'].hist()  
plt.show()  
print(dataset['Pclass'].isnull().values.any())

Survived_p1 = dataset.Survived[dataset['Pclass'] == 1].value_counts()
Survived_p2 = dataset.Survived[dataset['Pclass'] == 2].value_counts()
Survived_p3 = dataset.Survived[dataset['Pclass'] == 3].value_counts()

df=pd.DataFrame({'p1':Survived_p1, 'p2':Survived_p2, 'p3':Survived_p3})
print(df)
df.plot(kind='bar', stacked=True)
plt.title("survived by pclass")
plt.xlabel("pclass") 
plt.ylabel("count")
plt.show()
survived_S = dataset.Survived[dataset['Embarked'] == 'S'].value_counts()
survived_C = dataset.Survived[dataset['Embarked'] == 'C'].value_counts()
survived_Q = dataset.Survived[dataset['Embarked'] == 'Q'].value_counts()

print(survived_S)
df = pd.DataFrame({'S':survived_S, 'C':survived_C, 'Q':survived_Q})
df.plot(kind='bar', stacked=True)
plt.title("survived by embarked")
plt.xlabel("Embarked") 
plt.ylabel("count")
plt.show()
label=dataset.loc[:,'Survived']
data=dataset.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
testdat=testset.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

print(data.shape)
print(data.head())
def fill_nan(data):
    data_copy = data.copy(deep=True)
    data_copy['Age'] = data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy['Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy['Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy['Sex'] = data_copy['Sex'].fillna('male')
    data_copy['Embarked'] = data_copy['Embarked'].fillna('S')
    return data_copy
data_no_nan = fill_nan(data)
testdat_no_nan = fill_nan(testdat)
def transfer_sex(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Sex'] == 'female', 'Sex'] = 0
    data_copy.loc[data_copy['Sex'] == 'male', 'Sex'] = 1
    return data_copy

data_after_sex = transfer_sex(data_no_nan)
testdat_after_sex = transfer_sex(testdat_no_nan)
print(testdat_after_sex.head())
def transfer_embark(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 0
    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 1
    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 2
    return data_copy

data_after_embarked = transfer_embark(data_after_sex)
testdat_after_embarked = transfer_embark(testdat_after_sex)
print(testdat_after_embarked.head())
data_now = data_after_embarked
testdat_now = testdat_after_embarked
from sklearn.model_selection import train_test_split


train_data,test_data,train_labels,test_labels=train_test_split(data_now,label,random_state=0,test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
k_range = range(1, 51)
k_scores = []
for K in k_range:
    clf=KNeighborsClassifier(n_neighbors = K)
    clf.fit(train_data,train_labels)
    print('K=', K)
    predictions=clf.predict(test_data)
    score = accuracy_score(test_labels,predictions)
    print(score)
    # 检测模型precision， recall 等各项指标
    print(classification_report(test_labels, predictions))  
    print(confusion_matrix(test_labels, predictions))
    k_scores.append(score)
plt.plot(k_range, k_scores)
plt.xlabel('K for KNN')
plt.ylabel('Accuracy on validation set')
plt.show()
print(np.array(k_scores).argsort())
# 预测



# cross validation 找到最好的k值
from sklearn.model_selection import cross_val_score
clf=KNeighborsClassifier(n_neighbors=33)
scores = cross_val_score(clf, train_data, train_labels, cv=5)
print(scores)
from sklearn.model_selection import cross_val_score
clf=KNeighborsClassifier(n_neighbors=34)
scores = cross_val_score(clf, train_data, train_labels, cv=5)
print(scores)
from sklearn.model_selection import cross_val_score
clf=KNeighborsClassifier(n_neighbors=32)
scores = cross_val_score(clf, train_data, train_labels, cv=5)
print(scores)
# 预测
clf=KNeighborsClassifier(n_neighbors=33)
clf.fit(data_now,label)
result=clf.predict(testdat_now)
df = pd.DataFrame({"PassengerId": testset['PassengerId'],"Survived": result})
df.to_csv('submission.csv',header=True)
