import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
dataset=pd.read_csv('../input/train.csv')
testset=pd.read_csv('../input/test.csv')
dataset.columns
dataset.head()
print(dataset.dtypes)
print(dataset.describe())

Survived_m = dataset.Survived[dataset.Sex == 'male'].value_counts()
Survived_f = dataset.Survived[dataset.Sex == 'female'].value_counts()

df=pd.DataFrame({'male':Survived_m, 'female':Survived_f})
df.plot(kind='bar', stacked=True)
plt.title("survived by sex")
plt.xlabel("survived") 
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
plt.xlabel("pclass") 
plt.ylabel("count")
plt.show()
Survived_S = dataset.Survived[dataset['Embarked'] == 'S'].value_counts()
Survived_C = dataset.Survived[dataset['Embarked'] == 'C'].value_counts()
Survived_Q = dataset.Survived[dataset['Embarked'] == 'Q'].value_counts()

print(Survived_S)
df = pd.DataFrame({'S':Survived_S, 'C':Survived_C, 'Q':Survived_Q})
df.plot(kind='bar', stacked=True)
plt.title("Survived by Embarked")
plt.xlabel("Survival") 
plt.ylabel("count")
plt.show()
label=dataset.loc[:,'Survived']
data=dataset.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
test_data=testset.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

print(data.shape)
print(data)
def fill_NAN(data):  
    cp = data.copy(deep=True)
    cp['Age'] = cp['Age'].fillna(cp['Age'].median())
    cp['Fare'] = cp['Fare'].fillna(cp['Fare'].median())
    cp['Pclass'] = cp['Pclass'].fillna(cp['Pclass'].median())
    cp['Sex'] = cp['Sex'].fillna('female')
    cp['Embarked'] = cp['Embarked'].fillna('S')
    return cp


data_no_nan = fill_NAN(data)
test_data_no_nan = fill_NAN(test_data)

print(test_data.isnull().values.any())    
print(test_data_no_nan.isnull().values.any())
print(data.isnull().values.any())   
print(data_no_nan.isnull().values.any())    

print(data_no_nan)

print(data_no_nan['Sex'].isnull().values.any())

def transfer_sex(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Sex'] == 'female', 'Sex'] = 0
    data_copy.loc[data_copy['Sex'] == 'male', 'Sex'] = 1
    return data_copy

data_after_sex = transfer_sex(data_no_nan)
test_data_after_sex = transfer_sex(test_data_no_nan)
print(test_data_after_sex)
def transfer_embark(data):
    cp = data.copy(deep=True)
    cp.loc[cp['Embarked'] == 'S', 'Embarked'] = 0
    cp.loc[cp['Embarked'] == 'C', 'Embarked'] = 1
    cp.loc[cp['Embarked'] == 'Q', 'Embarked'] = 2
    return cp

data_after_embarked = transfer_embark(data_after_sex)
test_data_after_embarked = transfer_embark(test_data_after_sex)
print(test_data_after_embarked)
preprocessed_data = data_after_embarked
preprocessed_test_data = test_data_after_embarked
from sklearn.model_selection import train_test_split

train_set, validation_set, train_labels, validation_labels = train_test_split(preprocessed_data, label, random_state=0, test_size=0.2)
print(train_set.shape, validation_set.shape, train_labels.shape, validation_labels.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

k_range = range(1, 51)
k_scores = []
for k in k_range:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(train_set, train_labels)
    predictions = clf.predict(validation_set)
    score = accuracy_score(validation_labels, predictions)
    print(f'k={k}, score={score}')
    print(classification_report(validation_labels, predictions))
    k_scores.append(score)
# 预测

# 检测模型precision， recall 等各项指标

plt.plot(k_range, k_scores)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()
print(np.array(k_scores))
print(np.array(k_scores).argsort())
# 预测
clf = KNeighborsClassifier(n_neighbors=33)
clf.fit(preprocessed_data, label)
result = clf.predict(preprocessed_test_data)
print(result)
df = pd.DataFrame({'PassengerId': testset['PassengerId'], 'Survived': result})
df.to_csv('submission.csv', header=True, index=False)
