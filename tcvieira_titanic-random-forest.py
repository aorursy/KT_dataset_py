# Import modules
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split

# Figures inline and set visualization style
%matplotlib inline
sns.set()
os.listdir('../input')
# Import data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
print(df_train.shape)
df_train.head()
print(df_test.shape)
df_test.head()
df = df_train.append(df_test, sort=False)
df.info()
# Dealing with missing numerical variables
df['Age'] = df.Age.fillna(df.Age.median())
df['Fare'] = df.Fare.fillna(df.Fare.median())
df.info()
#df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
df['Surname'] = df['Name'].str.split(',').str[0]
df['Title'] = df['Name'].str.split(',').str[1].str.split().str[0]  
#df['Cabin Len'] = df.Cabin.str.split().str.len()
df['Cabin Letter'] = df['Cabin'].str[0]
df['Family_Size'] = df['SibSp'] + df['Parch']
df['Fare Per Person'] = df['Fare'] / (df['Family_Size'] + 1)
df['Number of Ticket Uses'] = df.groupby('Ticket', as_index=False)['Ticket'].transform(lambda s: s.count())
df['Average Fare per Person'] = df['Fare'] / df['Number of Ticket Uses'] 
for col in df.columns:  
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category')  # change text to category
        df[col] = df[col].cat.codes  # save code as column value
# RandomForest/Decision Tree it is interesting to replace NA by a value less then the minimum or greater then the maximum
#df.fillna(-1, inplace=True)
data_train = df.iloc[:891].copy()
data_test = df.iloc[891:].copy()
train, test = train_test_split(data_train, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_features=.5, random_state=42)
remove = ['Survived', 'PassengerId', 'Name', 'Cabin', 'Embarked']
feats = [col for col in df.columns if col not in remove]
rf.fit(train[feats], train['Survived'])
preds_train = rf.predict(train[feats])
preds = rf.predict(test[feats])
from sklearn.metrics import accuracy_score
accuracy_score(train['Survived'], preds_train)
accuracy_score(test['Survived'], preds)
rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, max_features=.5, random_state=42)
# train with training and test dataset
rf.fit(data_train[feats],data_train['Survived'])
preds_kaggle = rf.predict(data_test[feats])
submission = pd.DataFrame({ 'PassengerId': data_test['PassengerId'],
                            'Survived': preds_kaggle }, dtype=int)
submission.to_csv("submission.csv",index=False)
