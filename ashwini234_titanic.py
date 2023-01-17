# This Python 3 environment comes with many helpful analytics libraries installed
import pandas as pd
iowa_file_path = '../input/train.csv'
iowa_file_path2 = '../input/test.csv'
train = pd.read_csv(iowa_file_path)

test = pd.read_csv(iowa_file_path2)


train.head(3)
for df in [train, test]:
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])
for df in [train, test]:
    df['Sex_binary'] = df['Sex'].map({'male':1, 'female':0})
for df in [train, test]:    
    df['Embarkeds'] = df['Embarked'].map( {'Q': 0, 'S': 1, 'C': 2} ).astype(int)
for df in [train, test]:    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
for df in [train, test]:    
    df['IsAlone'] = 0
for df in [train, test]:    
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    
train['Age'] = train['Age'].fillna(25)
test['Age'] = test['Age'].fillna(25)
train['Fare'] = train['Fare'].fillna(40)
test['Fare'] = test['Fare'].fillna(40)
train['Embarked'] = train['Embarked'].fillna(1)
test['Embarked'] = train['Embarked'].fillna(1)
features = ['Pclass', 'Age', 'Sex_binary', 'Fare','Embarkeds', 'FamilySize']
target = ['Survived']
    
train[features].head(3)
train[target].head(3).values
from xgboost import XGBClassifier
clf = XGBClassifier(booster='dart')



clf.fit(train[features], train[target], verbose=False)

predictions = clf.predict(test[features])

predictions

   
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})

submission.head()
filename = 'cool2.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)