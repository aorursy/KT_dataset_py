import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/train.csv')
testset = pd.read_csv('../input/test.csv')

dataset.columns
print(type(dataset.columns))
print(dataset.dtypes)
print(dataset.describe())
print(type(dataset.Survived))
Survived_m = dataset.Survived[dataset.Sex =='male'].value_counts()
Survived_f = dataset.Survived[dataset.Sex =='female'].value_counts()

df=pd.DataFrame({'male':Survived_m, 'female':Survived_f})
print(df)
df.plot(kind='bar', stacked=True)
plt.title("survived by sex")
plt.xlabel("sex") 
plt.ylabel("count")
plt.show()
dataset['Age'].hist()  
plt.ylabel("Number") 
plt.xlabel("Age") 
plt.title('Age distribution')
plt.show() 

dataset[dataset.Survived==0]['Age'].hist()  
plt.ylabel("Number") 
plt.xlabel("Age") 
plt.title('Age distribution of people who did not survive')
plt.show()

dataset[dataset.Survived==1]['Age'].hist()  
plt.ylabel("Number") 
plt.xlabel("Age") 
plt.title('Age distribution of people who survived')
plt.show()
dataset['Fare'].hist()  
plt.ylabel("Number") 
plt.xlabel("Fare") 
plt.title('Fare distribution')
plt.show() 

dataset[dataset.Survived==0]['Fare'].hist()  
plt.ylabel("Number") 
plt.xlabel("Fare") 
plt.title('Fare distribution of people who did not survive')
plt.show()

dataset[dataset.Survived==1]['Fare'].hist()  
plt.ylabel("Number") 
plt.xlabel("Fare") 
plt.title('Fare distribution of people who survived')
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
plt.xlabel("Survived") 
plt.ylabel("count")
plt.show()
Survived_S = dataset.Survived[dataset['Embarked'] == 'S'].value_counts()
Survived_C = dataset.Survived[dataset['Embarked'] == 'C'].value_counts()
Survived_Q = dataset.Survived[dataset['Embarked'] == 'Q'].value_counts()

print(Survived_S)
df = pd.DataFrame({'S':Survived_S, 'C':Survived_C, 'Q':Survived_Q})
df.plot(kind='bar', stacked=True)
plt.title("survived by embarked")
plt.xlabel("Survived") 
plt.ylabel("count")
plt.show()
label=dataset.loc[:,'Survived']
data=dataset.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
testdat=testset.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

print(data.shape)
print(data)
print(testdat)
def fill_NAN(data):  
    data_copy = data.copy(deep=True)
    data_copy.loc[:,'Age'] = data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy.loc[:,'Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy.loc[:,'Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy.loc[:,'Sex'] = data_copy['Sex'].fillna('female')
    data_copy.loc[:,'Embarked'] = data_copy['Embarked'].fillna('S')
    return data_copy


data_no_nan = fill_NAN(data)
testdat_no_nan = fill_NAN(testdat)

print(testdat.isnull().values.any())    
print(testdat_no_nan.isnull().values.any())
print(data.isnull().values.any())   
print(data_no_nan.isnull().values.any())    

print(data_no_nan)
print(data_no_nan['Sex'].isnull().values.any())

def transfer_sex(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Sex'] == 'female', 'Sex'] = 0  # loc is to access a group of rows
    data_copy.loc[data_copy['Sex'] == 'male', 'Sex'] = 1
    return data_copy

data_after_sex = transfer_sex(data_no_nan)
testdat_after_sex = transfer_sex(testdat_no_nan)
print(testdat_after_sex)
    
def transfer_embark(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 0
    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 1
    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 2
    return data_copy

data_after_embarked = transfer_embark(data_after_sex)
testdat_after_embarked = transfer_embark(testdat_after_sex)
print(testdat_after_embarked)

data_now = data_after_embarked
testdat_now = testdat_after_embarked
from sklearn.model_selection import train_test_split


train_data,test_data,train_labels,test_labels=train_test_split(data_now,label,random_state=0,test_size=0.2)
print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
k_range = range(1, 51)
k_scores = []
for K in k_range:
    clf=KNeighborsClassifier(n_neighbors = K)
    clf.fit(train_data,train_labels)
    print('K=', K)
    predictions=clf.predict(test_data)
    score = accuracy_score(test_labels,predictions)
    print(score)
    k_scores.append(score)
plt.plot(k_range, k_scores)
plt.xlabel('K for KNN')
plt.ylabel('Accuracy on validation set')
plt.show()
print(np.array(k_scores).argsort())
clf=KNeighborsClassifier(n_neighbors=33)
clf.fit(data_now,label)
result=clf.predict(testdat_now)
print(result)
# df = pd.DataFrame({"PassengerId": testset['PassengerId'],"Survived": result})
# df.to_csv('./submission.csv',header=True)
# 这里的input file 是 read only， 所以就没有放在input里头了～

