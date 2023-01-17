import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
# Vamos iniciar o notebook importanto o Dataset
titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

# Podemos observar as primeiras linhas dele.
test_df.head()
print(test_df.shape, titanic_df.shape)
titanic_df.head()
titanic_df['Age'].isnull().any()
df_full = titanic_df.append(test_df)
age_median = df_full['Age'].median()
print(age_median)
titanic_df['Age'] = titanic_df['Age'].fillna(age_median)
test_df['Age'] = test_df['Age'].fillna(age_median)
titanic_df['Age'].isnull().any()
data = [titanic_df, test_df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'alone'] = 1
    dataset['alone'] = dataset['alone'].astype(int)
import re

data = [titanic_df, test_df]
deck = {"U": 1, "C": 2, "B": 3, "D": 4, "E": 5, "F": 6, "A": 7, "G": 8}

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

titanic_df['Cabin'].unique()
data = [titanic_df, test_df]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)

#titanic_df['Fare'].describe()
data = [titanic_df, test_df]
for dataset in data:
    dataset['AgeRange'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'AgeRange'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 22), 'AgeRange'] = 1
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 33), 'AgeRange'] = 2
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 44), 'AgeRange'] = 3
    dataset.loc[(dataset['Age'] > 44) & (dataset['Age'] <= 55), 'AgeRange'] = 4
    dataset.loc[(dataset['Age'] > 55) & (dataset['Age'] <= 66), 'AgeRange'] = 5
    dataset.loc[ dataset['Age'] > 66, 'AgeRange'] = 6
data = [titanic_df, test_df]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'FareBand'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'FareBand'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'FareBand']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'FareBand']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'FareBand']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'FareBand'] = 5
    dataset['FareBand'] = dataset['FareBand'].astype(int)
import seaborn as sns
sns.countplot(titanic_df['Sex']);
#from sklearn.preprocessing import LabelEncoder
#sex_encoder = LabelEncoder()

#sex_encoder.fit(list(titanic_df['Sex'].values) + list(test_df['Sex'].values))
#sex_encoder.classes_
#titanic_df['Sex'] = sex_encoder.transform(titanic_df['Sex'].values)
#test_df['Sex'] = sex_encoder.transform(test_df['Sex'].values)
#sns.countplot(titanic_df['Sex'], order=[1,0]);
titanic_df.head()
feature_names = ['Pclass', 'SibSp', 'Parch', 'Fare']
from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(titanic_df[feature_names].as_matrix(), 
                                                      titanic_df['Survived'].as_matrix(),
                                                      test_size=0.2,
                                                      random_state=42)
                                                      
                                                      
print(train_X.shape)
print(valid_X.shape)                                           
print(train_y.shape)
print(valid_y.shape)
from sklearn.ensemble import RandomForestClassifier

#Hiperparametros
rf_clf = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=7)

#Treino
rf_clf.fit(train_X, train_y)

print("Score Treino")
print(rf_clf.score(train_X, train_y))


print("Score Validação")
print(rf_clf.score(valid_X, valid_y))
import seaborn as sns
sns.barplot(rf_clf.feature_importances_, feature_names);
seed = 42

feature_names = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Age', 'Sex']

'''
X = titanic_df[feature_names].as_matrix()
y = titanic_df['Survived'].as_matrix()

from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(X,y, test_size=0.2,random_state=seed)
                                                                                                        
print(train_X.shape)
print(valid_X.shape)                                           
print(train_y.shape)
print(valid_y.shape)

rf_clf = RandomForestClassifier(random_state=seed, n_estimators=200, max_depth=5)
rf_clf.fit(train_X, train_y)

print(rf_clf.score(train_X, train_y))
print(rf_clf.score(valid_X, valid_y))

sns.barplot(rf_clf.feature_importances_, feature_names);
'''
titanic_df.head()['Name']
import re
def extract_title(name):
    x = re.search(', (.+?)\.', name)
    if x:
        return x.group(1)
    else:
        return ''
titanic_df['Name'].apply(extract_title).unique()
titanic_df['Title'] = titanic_df['Name'].apply(extract_title)
test_df['Title'] = test_df['Name'].apply(extract_title)
titanic_df.head()
#train_X.shape
titanic_df['Embarked']= titanic_df['Embarked'].fillna('Z')
#from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer

feature_names = ['Pclass', 'Deck', 'Sex', 'Age', 'AgeRange', 'SibSp', 'Parch', 'relatives', 'alone', 'Fare', 'FareBand', 'Title', 'Embarked' ]
dv = DictVectorizer()

dv.fit(titanic_df[feature_names].append(test_df[feature_names]).to_dict(orient='records'))
dv.feature_names_
from sklearn.model_selection import train_test_split

train_X, valid_X, train_y, valid_y = train_test_split(dv.transform(titanic_df[feature_names].to_dict(orient='records')),
                                                     titanic_df['Survived'],
                                                     test_size=0.2,
                                                     random_state=42)
train_X.shape
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#for n in range(1, 20):
rf_clf = RandomForestClassifier(random_state=seed, n_estimators=200, max_depth=5)
rf_clf.fit(train_X, train_y)

print(rf_clf.score(train_X, train_y))
print(rf_clf.score(valid_X, valid_y))

sns.barplot(rf_clf.feature_importances_, dv.feature_names_);
#test_df['Fare'] = test_df['Fare'].fillna(0)
test_df['Embarked']= test_df['Embarked'].fillna('Z')
test_X = dv.transform(test_df[feature_names].to_dict(orient='records'))
print(test_X.shape)
y_pred = rf_clf.predict(test_X)
y_pred.shape
submission_df = pd.DataFrame()
submission_df['PassengerId'] = test_df['PassengerId']
submission_df['Survived'] = y_pred
submission_df
submission_df.to_csv('submit_v5.csv', index=False)
