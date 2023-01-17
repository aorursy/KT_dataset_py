import pandas as pd
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
PassengerId = test['PassengerId']
train.tail()

full_data = [train,test]
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)
train.head()
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] ==1, 'IsAlone'] = 1
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
    dataset['CategoricalFare'] = pd.qcut(dataset['Fare'],4,labels = [1,2,3,4])
    dataset['CategoricalFare'] = dataset['CategoricalFare'].astype(int)

import numpy as np
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    dataset['CategoricalAge'] = pd.cut(dataset['Age'],5,labels = [1,2,3,4,5])
    dataset['CategoricalAge'] = dataset['CategoricalAge'].astype(int)
train.tail()
import re
def get_title(name):
    title_search = re.search('([A-Za-z]+)\.',name)
    if title_search:
        return title_search.group(1)
    return ""
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
train.head()
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt','Col','Don', 'Dr','Major','Rev','Sir,Jonkheer','Dona'],'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    dataset['Title'] = dataset['Title'].astype(int)
    
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train.head()
drop_elements = ["PassengerId","Name","Age","SibSp","Parch","Ticket","Fare","Cabin"]
train = train.drop(drop_elements,axis=1)
train.head()
test = test.drop(drop_elements,axis=1)

import seaborn as sns
import matplotlib.pyplot as plt
colormap = plt.cm.RdBu
plt.figure(figsize=(14,20))
sns.heatmap(train.astype(float).corr(),cmap=colormap,square=True,annot=True)
from sklearn.linear_model import LogisticRegression
X_train = train.drop("Survived",axis=1)
Y_train = train["Survived"]

lr = LogisticRegression()
lr.fit(X_train,Y_train)
acc = round(lr.score(X_train,Y_train) *100,2)
Y_pred = lr.predict(test)

submission = pd.DataFrame({
    "PassengerId":PassengerId,
    "Survived":Y_pred
})
submission.to_csv("sumbssion.csv",index=False)




