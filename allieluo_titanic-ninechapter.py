import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/train.csv')
testset = pd.read_csv('../input/test.csv')
dataset.head()
print(dataset.dtypes)
dataset.describe()
#observe data Sex, not number
Survived_m = dataset.Survived[dataset.Sex=='male'].value_counts()
Survived_f = dataset.Survived[dataset.Sex=='female'].value_counts()

df = pd.DataFrame({'male':Survived_m, 'female':Survived_f})
df.plot(kind='bar',stacked=True)
plt.title('Survived number by Sex')
plt.xlabel('if survived')
plt.ylabel('Number of people')
plt.show()
dataset['Age'].hist()
plt.title('Total number by Age')
plt.xlabel('Age')
plt.ylabel('Number')
plt.show()

# Survived
dataset[dataset.Survived==1]['Age'].hist()
plt.title('Survived number by Age')
plt.xlabel('Age')
plt.ylabel('Number')
plt.show()

# Not Survived
dataset[dataset.Survived==0]['Age'].hist()
plt.title('Not Survived number by Age')
plt.xlabel('Age')
plt.ylabel('Number')
plt.show()

dataset['Fare'].hist()
plt.title('Total number by Fare')
plt.xlabel('Fare')
plt.ylabel('Number')
plt.show()

# Survived
dataset[dataset.Survived==1]['Fare'].hist()
plt.title('Survived number by Fare')
plt.xlabel('Fare')
plt.ylabel('Number')
plt.show()

# Not Survived
dataset[dataset.Survived==0]['Fare'].hist()
plt.title('Not Survived number by Fare')
plt.xlabel('Fare')
plt.ylabel('Number')
plt.show()

dataset['Pclass'].hist()
plt.title('Total number by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Number')
plt.show()

# Survived
dataset[dataset.Survived==1]['Pclass'].hist()
plt.title('Survived number by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Number')
plt.show()

# Not Survived
dataset[dataset.Survived==0]['Pclass'].hist()
plt.title('Not Survived number by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Number')
plt.show()

## another way to look at Pclass
dataset['Pclass'].hist()
plt.show()
print(dataset['Pclass'].isnull().values.any())

Survived_p1 = dataset.Survived[dataset['Pclass']==1].value_counts()
Survived_p2 = dataset.Survived[dataset['Pclass']==2].value_counts()
Survived_p3 = dataset.Survived[dataset['Pclass']==3].value_counts()
df = pd.DataFrame({'p1':Survived_p1,'p2':Survived_p2,'p3:':Survived_p3})
print(df)
df.plot(kind='bar',stacked=True)
plt.title('Survived by pClass')
plt.xlabel('pClass')
plt.ylabel('number')
# another way
Survived_p1 = dataset.Survived[dataset.Pclass==1].value_counts()
Survived_p2 = dataset.Survived[dataset.Pclass==2].value_counts()
Survived_p3 = dataset.Survived[dataset.Pclass==3].value_counts()
df = pd.DataFrame({'p1':Survived_p1,'p2':Survived_p2,'p3:':Survived_p3})
print(df)
df.plot(kind='bar',stacked=True)
plt.title('Survived by pClass')
plt.xlabel('pClass')
plt.ylabel('number')
##观察登船地点 Embarked
dataset.Embarked.head(20)
SurvivedEmbarked_S = dataset.Survived[dataset['Embarked']=='S'].value_counts()
SurvivedEmbarked_C = dataset.Survived[dataset['Embarked']=='C'].value_counts()
SurvivedEmbarked_Q = dataset.Survived[dataset['Embarked']=='Q'].value_counts()
print(SurvivedEmbarked_S)
df = pd.DataFrame({'S':SurvivedEmbarked_S,'C':SurvivedEmbarked_C,'Q':SurvivedEmbarked_Q})
df.plot(kind='bar',stacked=True)
plt.title('Survived by Embarked place')
plt.xlabel('if survived')
plt.ylabel('number')

#keep these features: Sex, Age, Fare, Pclass, Embarked
#separate these features from dataset
label=dataset.loc[:,'Survived'] #label - classification problem
training_data=dataset.loc[:,['Pclass','Sex','Age','Fare','Embarked']]
test_data=testset.loc[:,['Pclass','Sex','Age','Fare','Embarked']]

#print the data
print(training_data.shape)
print(training_data.head())
label.head()
#replace NaN
def fill_NaN(useful_feature_data):
    data_copy = useful_feature_data.copy(deep=True)#deep copy, make a new copy, do not modify the original data
    # fill in NaN for each feature in the copy and return the copy
    # use python function fillna
    data_copy.loc[:,'Age']=data_copy['Age'].fillna(data_copy['Age'].median())# replace number with median
    data_copy.loc[:,'Fare']=data_copy['Fare'].fillna(data_copy['Fare'].median())# replace number with median
    data_copy.loc[:,'Pclass']=data_copy['Pclass'].fillna(data_copy['Pclass'].median())# replace number with median
    data_copy.loc[:,'Sex']=data_copy['Sex'].fillna('male')# replace Sex with majority 'male'
    data_copy.loc[:,'Embarked']=data_copy['Embarked'].fillna('S')# replace Embarked with majority 'S'
    return data_copy

training_data_elimilate_nan = fill_NaN(training_data)
test_data_elimilate_nan = fill_NaN(test_data)
# check before and after to see if NaN is elimilated
print(training_data.isnull().values.any())
print(test_data.isnull().values.any())
print(training_data_elimilate_nan.isnull().values.any())
print(test_data_elimilate_nan.isnull().values.any())
#print contents of useful_feature_data_elimilate_nan
print(training_data_elimilate_nan.head())
#transform Sex into 0 or 1
def transform_Sex(data):
    data_copy = data.copy(deep=True)#make a new copy to keep original data intact
    data_copy.loc[data_copy['Sex']=='female','Sex']=0 #loc can access a group of rows and columns by labels
    # first label is Boolean, only select rows with True
    data_copy.loc[data_copy['Sex']=='male','Sex']=1
    return data_copy

data_after_transform_Sex = transform_Sex(training_data_elimilate_nan)
testdata_after_transform_Sex = transform_Sex(test_data_elimilate_nan)
print(testdata_after_transform_Sex.head())
data_after_transform_Sex.dtypes
#transform Embarked into 0 or 1
def transform_Embarked(data):
    data_copy = data.copy(deep=True)#make a new copy to keep original data intact
    data_copy.loc[data_copy['Embarked']=='S','Embarked']=0 #loc can access a group of rows and columns by labels
    data_copy.loc[data_copy['Embarked']=='C','Embarked']=1
    data_copy.loc[data_copy['Embarked']=='Q','Embarked']=2
    return data_copy

data_after_transform_Embarked = transform_Embarked(data_after_transform_Sex)
testdata_after_transform_Embarked = transform_Embarked(testdata_after_transform_Sex)
print(testdata_after_transform_Embarked.head())
## train data
data_for_training = data_after_transform_Embarked
testdata_now = testdata_after_transform_Embarked
from sklearn.model_selection import train_test_split #Split arrays or matrices into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(data_for_training, label, random_state=0,test_size=0.2)
# check shape
X_train.shape, y_train.shape
X_train.head()
y_train.head(20)#labels
X_test.shape, y_test.shape
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
k_range = range(1,51) #try k = 1, 2, ..., 51
k_scores = []
for K in k_range:
    clf = KNeighborsClassifier(n_neighbors = K)
    clf.fit(X_train, y_train)
    print('K=',K)
    predictions=clf.predict(X_test)
    score=accuracy_score(y_test,predictions)
    print(score)
    k_scores.append(score)
    
#find k which has the highest score
plt.plot(k_range,k_scores)#manual inspection
plt.xlabel('K for KNN')
plt.ylabel('Accuracy on validation sets')
plt.show()
print(np.array(k_scores).argsort())#Returns the indices that would sort an array in ascending order.
k_best = np.array(k_scores).argsort()[-1]+1;
print('The highest score is obtained when k=',k_best)
from sklearn.model_selection import cross_val_score
k_range = range(1,51) #try k = 1, 2, ..., 51
k_scores = []
for K in k_range:
    clf = KNeighborsClassifier(n_neighbors = K)
    k_scores.append(cross_val_score(clf,X_train, y_train,cv=5).mean())
    
k_best = np.array(k_scores).argsort()[-1]+1
print("the best K=",k_best,"best average accuracy with this K=",max(k_scores))
#predict with k_best
clf=KNeighborsClassifier(n_neighbors=k_best)
clf.fit(data_for_training,label)
result=clf.predict(testdata_now)
print(result)
df = pd.DataFrame({"PassengerId": testset['PassengerId'],"Survived": result})
df.to_csv('submission.csv',header=True, index=False)
