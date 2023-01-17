import os
import pandas as pd
os.chdir("path/directory")
df_train = pd.read_csv('train.csv', header=0, index_col='PassengerId')
df_test = pd.read_csv('test.csv', header=0, index_col='PassengerId')
df = pd.concat([df_train, df_test], keys=['train', 'test'], sort=False)
df.head()
df.describe()
df.isnull().sum()
sns.heatmap(df.isnull(), cbar=False)
sns.boxplot(x=df['Age'])
sns.boxplot(x=df['SibSp'])
sns.boxplot(x=df['Parch'])
sns.boxplot(x=df['Fare'])
df['Title'] = df['Name'].apply(lambda name: name[name.index(',') + 2 : name.index('.')])
df['FamilySize'] = (df['SibSp'] + df['Parch'] + 1)
df.FamilySize= df.FamilySize.astype(float)
df.Pclass = df.Pclass.astype(float)
print(df.Title.value_counts())
ReducedTitles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Sir",
    "Sir" :       "Sir",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Sir",
    "Lady" :      "Royalty"
}

df.Title = df.Title.map(ReducedTitles)

print(df.Title.value_counts())
df.drop(columns=['Cabin'], inplace=True) #cabin has too many missing
df.loc['train'].Embarked.mode()
df.Embarked.fillna("S", inplace=True)
groupby_Pclass = df.loc['train'].Fare.groupby(df.loc['train'].Pclass)
groupby_Pclass.mean()
df.Fare.fillna(13.302889, inplace=True)
#df['Age'].fillna(df.loc['train'].Age.median(),inplace=True)
median_age_by_title = pd.DataFrame(df.groupby('Title')['Age'].median())
median_age_by_title.rename(columns = {'Age': 'MedianAgeByTitle'}, inplace=True)
df = df.merge(median_age_by_title, left_on='Title', right_index=True)
df.Age.fillna(df.MedianAgeByTitle, inplace=True)
df.drop(columns=['MedianAgeByTitle'], inplace=True)
df.isnull().sum()
import numpy as np
from sklearn_pandas import DataFrameMapper as DFM
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.grid_search import GridSearchCV as KCV
from sklearn import svm

df.drop(columns=['Parch', 'SibSp', 'Name','Ticket'], inplace=True)
#separo train e test, dati e target
train_df, test_df = df.loc['train'], df.loc['test']
#adjust train set
train_predvalues = train_df.pop('Survived')
train_data = train_df

#adjust test set
test_data = test_df.drop(columns=['Survived'])
test_IDs = test_df.index.values

mapper = DFM([(['Age', 'Fare', 'Pclass'], StandardScaler()),
              ('Sex'                , LabelBinarizer()), 
              ('Embarked'           , LabelBinarizer()),
              ('Title'              , LabelBinarizer())],
             default=None,
             df_out=True)

train_data = mapper.fit_transform(train_data)
test_data = mapper.transform(test_data)
param_grid = {
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma' : [0.001, 0.01, 0.1, 1]
}

grid_svc = KCV(svm.SVC(), param_grid, cv=10, scoring='accuracy')
grid_svc.fit(train_data, train_predvalues)

print('Best score: {}'.format(grid_svc.best_score_))
print('Best parameters: {}'.format(grid_svc.best_params_))
svc = svm.SVC(**grid_svc.best_params_).fit(train_data, train_predvalues)
#preparo sottomissione
res = pd.DataFrame({'PassengerId': test_IDs,
                    'Survived'   : svc.predict(test_data).astype(int)})

res.to_csv('path/results.csv', index=False)




