import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn
print (sklearn.__version__)
# suppress warnings
import warnings  
warnings.filterwarnings('ignore')
train_df = pd.read_csv("../input/train.csv")
test_df  = pd.read_csv("../input/test.csv")
# use both datasets for inspection by saving them in a new dataframe df
df = pd.concat([train_df,test_df], axis=0, ignore_index=False) 
print('Size of training set: {:d}'.format(len(train_df)))
print('Size of test set: {:d}'.format(len(test_df)))
df.head(10)
train_df.describe()
# print summary of missing fields in the training set
print("Training set\n")
print(train_df.isnull().sum(axis=0))
print("\n")
print("Train and test set\n")
print(df.isnull().sum(axis=0))
# drop Cabin 
train_df = train_df.drop(['Cabin'], axis=1)
df = df.drop(['Cabin'], axis=1)
print('Missing values per column in %')
print(((1 - train_df.count()/len(train_df.index))*100).apply(lambda x: '{:.1f}%'.format(x)))
# look at the row containing awkward minimum value (.42) for Age
train_df.loc[train_df['Age'].idxmin(axis=1)]
# look at a few records with missing Age
train_df[train_df.Age.isnull()].head()
train_df['Age'] = train_df.groupby(['Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))
df['Age'] = df.groupby(['Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))
# check -- now there should be no rows with null Age fields
train_df[train_df.Age.isnull()].head()
train_df['Fare'] = train_df.groupby(['Pclass'])['Fare'].transform(lambda x: x.fillna(x.mean()))
df['Fare'] = df.groupby(['Pclass'])['Fare'].transform(lambda x: x.fillna(x.mean()))
# check -- now there should be no rows with null Age fields
train_df[train_df.Fare.isnull()].head()
# just two records with missing `Embarked` data
print('Number of records with missing Embarked data: {}'.format(len(train_df[train_df.Embarked.isnull()])))

# check occurrences of different types of embarkment
train_df['Embarked'].value_counts(dropna=False)
# fill the two missing values with the most frequent value, which is "S".
train_df["Embarked"] = train_df["Embarked"].fillna("S")
df['Embarked'].value_counts(dropna=False)
# any more nans?
len(train_df[pd.isnull(train_df).any(axis=1)])
train_df[train_df['Survived']==0]
train_df[train_df['Survived']==0]
# add `FirstName` and `LastName` columns that will be needed later
train_df['LastName'],train_df['FirstName'] = train_df['Name'].str.split(',', 1).str
df['LastName'],df['FirstName'] = df['Name'].str.split(',', 1).str
# foreign names
train_df['Foreign'] = False
train_df['Foreign'] = train_df['LastName'].str.endswith(("ic", "sson", "ff", "i", "o", "u", "ski", "a"))
df['Foreign'] = False
df['Foreign'] = df['LastName'].str.endswith(("ic", "sson", "ff", "i", "o", "u", "ski", "a"))
train_df[train_df['Foreign']].head(10)
# are names ending in "ff" Russian?
# how many of them survived?
train_df[['FirstName', 'LastName', 'Survived']][train_df['LastName'].str.endswith("ff")].head()
# none survived
train_df[train_df['LastName'].str.endswith("ff") & train_df['Survived']>0]
# also for names ending in "ic" there are very few (1/20) survivors
train_df[train_df['LastName'].str.endswith("ic")]['Survived'].value_counts()
train_df['Name_len'] = train_df['Name'].apply(lambda x: len(x)).astype(int)
train_df['Name_end'] = train_df['LastName'].str[-1:]
df['Name_len'] = df['Name'].apply(lambda x: len(x)).astype(int)
df['Name_end'] = df['LastName'].str[-1:]
# ADD FEATURES
print(df.columns)
dummy_features=['Age', 'Embarked', 'Fare', 'Name', 'Parch', 'Pclass', 'Sex', 'SibSp',
       'Survived', 'Ticket', 'LastName', 'FirstName', 
       'Name_len', 'Foreign']

df_dummies = df[dummy_features]
df = pd.get_dummies(df[dummy_features])
train_features = df.iloc[:891,:]
train_labels = train_features.pop('Survived').astype(int)
test_features = df.iloc[891:,:].drop('Survived',axis=1)
print(df.columns)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
models=[KNeighborsClassifier(), LogisticRegression(), GaussianNB(), SVC(), DecisionTreeClassifier(),
        RandomForestClassifier(), GradientBoostingClassifier(), AdaBoostClassifier()]
names=['KNN', 'LR', 'NB', 'SVM', 'Tree', 'RF', 'GB', 'Ada']
for name,model in zip(names, models):
    score = cross_val_score(model, train_features, train_labels, cv=5)
    print('{} :: {} , {}'.format(name, score.mean(), score))
models=[LogisticRegression(),RandomForestClassifier()]
names=['LR','RF']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_features_scaled = scaler.fit(train_features).transform(train_features)
test_features_scaled = scaler.fit(test_features).transform(test_features)
for name,model in zip(names,models):
    score = cross_val_score(model,train_features_scaled,train_labels,cv=5)
    print('{} :: {} , {}'.format(name,score.mean(),score))
# Initialize the model class
model = RandomForestClassifier()

# Train the algorithm using all the training data
model.fit(train_features,train_labels)

# Make predictions using the test set.
predictions = model.predict(test_features)

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": test_df['PassengerId'],
        "Survived": predictions
    })

# uncomment to save submission file
# submission.to_csv("./output/RandomForestClassifierNormName.csv", index=False)
