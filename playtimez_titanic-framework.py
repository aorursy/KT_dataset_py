import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
train= pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.shape, test.shape
#train.info()
train.columns
train.dtypes
train.describe()
# Check if there is missing data in datasset
train.isnull().values.any(), test.isnull().values.any()
train.head()
male_s = train.Survived[train['Sex'] == 'male'].value_counts()
female_s = train.Survived[train['Sex'] == 'female'].value_counts()

df_sex = pd.DataFrame({'m': male_s, 'f': female_s})
df_sex.plot(kind ='bar', stacked = True)
plt.subplot(131)
train['Age'].hist(density = True)
plt.subplot(132)
train.Age[train['Survived'] == 0].hist(density = True)
plt.subplot(133)
train.Age[train['Survived'] == 1].hist(density = True)
plt.subplot(131)
plt.ylim(0,0.03)
train['Fare'].hist(density = True)
plt.ylim(0,0.03)
plt.subplot(132)
train.Fare[train['Survived'] == 0].hist(density = True)
plt.subplot(133)
plt.ylim(0,0.03)
train.Fare[train['Survived'] == 1].hist(density = True)
P1_s = train.Survived[train['Pclass'] == 1].value_counts()
P2_s = train.Survived[train['Pclass'] == 2].value_counts()
P3_s = train.Survived[train['Pclass'] == 3].value_counts()

df_pclass = pd.DataFrame({'1': P1_s, '2': P2_s, '3': P3_s})
df_pclass.plot(kind ='bar', stacked = True)
S_s = train.Survived[train['Embarked'] == 'S'].value_counts()
C_s = train.Survived[train['Embarked'] == 'C'].value_counts()
Q_s = train.Survived[train['Embarked'] == 'Q'].value_counts()

df_embarked = pd.DataFrame({'S': S_s, 'C': C_s, 'Q': Q_s})
df_embarked.plot(kind ='bar', stacked = True)
train_select = train.loc[:,['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
#train_select.head()
test_select=test.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
train_select.head()
def fill_NA(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[:, 'Age'] = data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy.loc[:,'Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy.loc[:,'Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy.loc[:,'Sex'] = data_copy['Sex'].fillna('female')
    data_copy.loc[:,'Embarked'] = data_copy['Embarked'].fillna('S')
    return data_copy

train_filled = fill_NA(train_select)
test_filled = fill_NA(test_select)
train_filled.head()
def sex2num(data):
    data_copy = data.copy(deep = True)
    #data_copy.Sex[data_copy['Sex'] == 'female'] = 0
    #data_copy.Sex[data_copy['Sex'] == 'male'] = 1
    data_copy.loc[data_copy['Sex'] == 'female', 'Sex'] = 0
    data_copy.loc[data_copy['Sex'] == 'male', 'Sex'] = 1
    
    return data_copy
    
train_sex2num = sex2num(train_filled)
test_sex2num = sex2num(test_filled)

train_sex2num.head()
    
def embark2num(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 0
    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 1
    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 2
    return data_copy
    
train_embark2num = embark2num(train_sex2num)
test_embark2num = embark2num(test_sex2num)

train_embark2num.columns
    
train_data = train_embark2num
X = train_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
#y = train_data[['Survived']]
y = train_data.loc[:,'Survived']
test_data = test_embark2num

from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,random_state=0,test_size=0.2)
print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
k_range = range(1, 51)
k_scores = []
for K in k_range:
    clf=KNeighborsClassifier(n_neighbors = K)
    clf.fit(Xtrain,ytrain)
    print('K=', K)
    predictions=clf.predict(Xtest)
    score = accuracy_score(ytest,predictions)
    print(score)
    k_scores.append(score)
    
# 预测
plt.plot(k_range, k_scores)    
plt.xlabel('K for KNN')
plt.ylabel('Accuracy on validation set')
plt.show()
print(np.array(k_scores).argsort())

# 预测
clf=KNeighborsClassifier(n_neighbors=33)
clf.fit(X,y)
y_pred = clf.predict(X)
# 检测模型precision， recall 等各项指标
from sklearn.metrics import classification_report
print(classification_report(y, y_pred))
result=clf.predict(test_data)
df = pd.DataFrame({"PassengerId": test['PassengerId'],"Survived": result})
df.to_csv('submission.csv',header=True, index=False)
