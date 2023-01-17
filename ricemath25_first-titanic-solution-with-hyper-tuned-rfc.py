import numpy as np
import os
import pandas as pd
import sklearn.ensemble
import sklearn.model_selection
train_df = pd.read_csv('../input/train.csv')
train_df.info()
train_df.head()
test_df = pd.read_csv('../input/test.csv')
print(test_df.shape)
train_df['Survived'].value_counts(normalize=True)
train_df.corr()["Survived"]
train_df.fillna(train_df.mean(axis=0), axis=0).corr()["Survived"]
train_df['Pclass'].value_counts()
train_df.groupby(['Pclass'])['Survived'].mean()
train_df['Name'].apply(lambda name: len(name)).hist()
list(train_df['Name'])[:20]
train_df['Sex'].value_counts()
train_df.groupby(['Sex'])['Survived'].mean()
train_df['Age'].hist()
train_df.groupby(pd.qcut(train_df['Age'], 3))['Survived'].mean()
train_df['SibSp'].hist()
train_df['Parch'].hist()
train_df['Fare'].hist()
train_df.groupby(pd.qcut(train_df['Fare'], 3))['Survived'].mean()
train_df['Cabin'].isnull().value_counts()
train_df['Embarked'].value_counts()
def process(df):
    filt_df = df[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin']].copy()
    filt_df.columns = ['ID', 'class', 'name_length', 'is_female', 'age', 'sibling_count', 'parent_child_count', 'price', 'origin', 'has_cabin']
    
    filt_df['name_length'] = filt_df['name_length'].apply(lambda name: len(name))
    filt_df['is_female'] = filt_df['is_female'].apply(lambda sex: 1 if sex == 'female' else 0)
    
    filt_df = pd.concat([filt_df[[col for col in filt_df if col!='origin']], pd.get_dummies(filt_df['origin'])], axis=1)
    filt_df_columns = list(filt_df.columns)
    filt_df_columns[-3:] = ['origin_' + col.lower() for col in filt_df_columns[-3:]]
    filt_df.columns = filt_df_columns
    
    filt_df['has_family'] = filt_df['sibling_count'] + filt_df['parent_child_count']
    filt_df['has_family'] = filt_df['has_family'].apply(lambda count: 1 if count > 0 else 0)
    filt_df.drop(columns=['sibling_count', 'parent_child_count'])
    
    filt_df['has_cabin'] = filt_df['has_cabin'].notnull().astype('int')
        
    filt_df = filt_df.fillna(filt_df.mean(axis=0), axis=0)
    
#     filt_df['name_length'] = (filt_df['name_length'] - filt_df['name_length'].mean()) / filt_df['name_length'].std()
#     filt_df['age'] = (filt_df['age'] - filt_df['age'].mean()) / filt_df['age'].std()
#     filt_df['price'] = (filt_df['price'] - filt_df['price'].mean()) / filt_df['price'].std()
    
    return(filt_df)
train_filt_df = process(train_df)
train_filt_df = pd.concat([train_filt_df, train_df['Survived']], axis=1)
train_filt_df_columns = list(train_filt_df.columns)
train_filt_df_columns[-1] = 'is_survived'
train_filt_df.columns = train_filt_df_columns
train_filt_df.head()
model = sklearn.ensemble.RandomForestClassifier(random_state=1, n_jobs=-1)
param_grid = {'criterion': ['gini', 'entropy'], 'bootstrap': [True, False], 'max_features': [3, 6, 9, 12], 'n_estimators': [3, 10, 30, 100]}
scoring_metric='accuracy'
cross_val_splits=4

train_X_arr = np.array(train_filt_df[[col for col in train_filt_df if col not in ['ID', 'is_survived']]])
train_Y_arr = np.array(train_filt_df['is_survived'])

grid_search = sklearn.model_selection.GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring_metric, cv=cross_val_splits)
grid_search.fit(train_X_arr, train_Y_arr)
model = grid_search.best_estimator_
model.oob_score=True
model
model.fit(train_X_arr, train_Y_arr)
scores = np.sqrt(sklearn.model_selection.cross_val_score(model, train_X_arr, train_Y_arr, scoring=scoring_metric, cv=cross_val_splits))
print('Scores: {0}'.format([round(x, 2) for x in scores]))
print('Mean: {0}'.format(round(scores.mean(), 2)))
print('STD: {0}'.format(round(scores.std(), 2)))
print('OOB Score: {0}'.format(round(model.oob_score_, 2)))
test_filt_df = process(test_df)
test_X_arr = np.array(test_filt_df[[col for col in test_filt_df if col not in ['ID', 'is_survived']]])
predictions = model.predict(test_X_arr)
print(len(predictions))
predictions[:10]
results_df = pd.DataFrame(list(test_filt_df['ID']))
results_df['predictions'] = list(predictions)
results_df.columns = ['PassengerId', 'Survived']
# results_df.to_csv('results.csv', header=True, index=False)
results_df.head()
