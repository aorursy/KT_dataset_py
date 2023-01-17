import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head(4)
full_data = [train, test]
#transform to logistic value
train['Has_Cabin'] = train['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    

train.loc[ train['Fare'] <= 7.91, 'Fare']  = 0
train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1
train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare']   = 2
train.loc[ train['Fare'] > 31, 'Fare'] = 3
train['Fare'] = train['Fare'].astype(int)
    

train.loc[ train['Age'] <= 14, 'Age'] = 0
train.loc[(train['Age'] > 14) & (train['Age'] <= 30), 'Age'] = 1
train.loc[(train['Age'] > 30) & (train['Age'] <= 50), 'Age'] = 2
train.loc[(train['Age'] > 50) & (train['Age'] <= 70), 'Age'] = 3
train.loc[ train['Age'] > 70, 'Age'] = 4 ;
train = train.dropna()
test = test.dropna()
sns.catplot(x="Sex", y="Survived", hue="Embarked", kind="bar", data=train);
sns.catplot(x="Sex", y="Survived", hue="IsAlone", kind="bar", data=train)
sns.catplot(x="Age", y = "Survived", kind="bar", data=train)
sns.catplot(x="Sex", y = "Survived", hue = "Age", kind="bar", data=train)
sns.catplot(x="Fare", y = "Survived", kind="bar", data=train)
sns.catplot(x="Sex", y = "Survived", hue = "Fare", kind="bar", data=train)
logreg = LogisticRegression()
features = ['Age','Sex','SibSp','Parch','Fare']
logreg.fit(train[features],train['Survived'])
predictions = logreg.predict(test[features])

pred_result = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})
pred_result.head()