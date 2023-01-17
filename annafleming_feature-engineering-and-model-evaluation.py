import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
%matplotlib inline
from datetime import datetime
sns.set()
import warnings
warnings.filterwarnings('ignore')
original_train_df = pd.read_csv('../input/train.csv')
original_test_df = pd.read_csv('../input/test.csv')
original_train_df.shape, original_test_df.shape
combined_df = original_train_df.append(original_test_df)
combined_df.head()
original_train_df['Survived'].value_counts()
combined_df = combined_df.drop(['PassengerId', 'Ticket'], axis=1)
sns.kdeplot(combined_df[combined_df.Survived == 1]['Age'].dropna(), label='Survived', shade=True)
sns.kdeplot(combined_df[combined_df.Survived == 0]['Age'].dropna(), label='Died', shade=True)
combined_df['Cabin'] = combined_df['Cabin'].fillna('U').astype(str)
combined_df['Deck'] = combined_df['Cabin'].apply(lambda x: x[0])
combined_df = combined_df.drop('Cabin', axis=1)
sns.countplot(y="Deck", hue="Survived", data=combined_df)
combined_df["Embarked"].value_counts()
sns.countplot(y="Embarked", hue="Survived", data=combined_df)
sns.kdeplot(combined_df[combined_df.Survived == 1]['Fare'].dropna(), label='Survived', shade=True)
sns.kdeplot(combined_df[combined_df.Survived == 0]['Fare'].dropna(), label='Died', shade=True)
def extract_prefix(name):
    return name.split(',')[1].split('.')[0].strip()

combined_df['Prefix'] = combined_df['Name'].apply(extract_prefix)

combined_df = combined_df.drop('Name', axis=1)

combined_df['Prefix'].value_counts()
prefix_mapping = {'Ms': 'Miss', 
                  'Mlle': 'Miss', 
                  'Mme': 'Mrs', 
                  'Col': 'Sir',
                  'Major': 'Sir', 
                  'Dona' : 'Lady', 
                  'the Countess': 'Lady',
                  'Capt': 'Sir',  
                  'Don': 'Sir',  
                  'Jonkheer': 'Sir'}
combined_df['Prefix'] = combined_df['Prefix'].replace(prefix_mapping)
combined_df['Prefix'].value_counts()
sns.countplot(y="Prefix", hue="Survived", data=combined_df)
combined_df['Parch'].value_counts()
combined_df['SibSp'].value_counts()
(combined_df['SibSp'] + combined_df['Parch']).value_counts()
sns.countplot(y="Pclass", hue="Survived", data=combined_df)
sns.countplot(y="Sex", hue="Survived", data=combined_df)
combined_df.isnull().sum()
median_ages = combined_df[['Age', 'Prefix']].groupby('Prefix').median()
median_ages
for title in median_ages.index:
    title_mask = (combined_df['Prefix'] == title) & (combined_df['Age'].isnull()) 
    median_value = float(median_ages.loc[title])
    combined_df.loc[title_mask,'Age'] = median_value
combined_df["Embarked"] = combined_df["Embarked"].fillna('S')
combined_df[combined_df['Fare'].isnull()]
median_fare = combined_df[(combined_df['Embarked'] == 'S') & (combined_df['Pclass'] == 3)]['Fare'].median()
print(median_fare)
combined_df["Fare"] = combined_df["Fare"].fillna(median_fare)
# Age
bins_count = 7
combined_df['Age_Bins'] = pd.qcut(combined_df['Age'], bins_count, labels=list(range(1,bins_count + 1)))

# Fare
bins_count = 5
combined_df['Fare_Bins'] = pd.qcut(combined_df['Fare'], bins_count, labels=list(range(1,bins_count + 1)))

combined_df = combined_df.drop(['Age', 'Fare'], axis=1)
categorical_columns = ['Embarked', 'Pclass', 'Sex', 'Deck', 'Prefix', 'Age_Bins', 'Fare_Bins']
combined_df = pd.get_dummies(combined_df, columns=categorical_columns, drop_first=True)
combined_df.columns
train_df = combined_df[~combined_df['Survived'].isnull()]
test_df = combined_df[combined_df['Survived'].isnull()]
train_df['Survived'] = train_df['Survived'].astype(int)
test_df = test_df.drop('Survived', axis=1)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
def test_models(df, label, models, scoring):
    num_folds = 5
    seed = 7
    X = df.drop(label, axis=1).values
    y = df[label].values
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=.30, random_state=seed)
    results = {}
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed)
        results[name] = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        
    for name in results:
        print("%s: %s" % (name, results[name].mean()))
        
    return results
def show_model_comparison_plot(results):
    fig = plt.figure() 
    fig.suptitle('Algorithm Comparison') 
    ax = fig.add_subplot(111) 
    plt.boxplot(results.values()) 
    ax.set_xticklabels(results.keys())
    ax.set_ylim(0,1)
    plt.show()
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier())) 
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC())) 
results = test_models(train_df, 'Survived', models , 'accuracy')
show_model_comparison_plot(results)
def param_tuning(df,label, model, scoring, parameters):
    num_folds = 5
    seed = 7
    X = df.drop(label, axis=1).values
    y = df[label].values
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=.30, random_state=seed)
    grid = GridSearchCV(model, parameters, cv=5)
    grid.fit(X_train, Y_train)
    return grid.best_params_, grid.best_score_
params_svc = {'svc__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
              'svc__C': [0.001, 0.01, 0.1, 1, 10, 100]}
model = make_pipeline(SVC())
param_tuning(train_df, 'Survived', model , 'accuracy', params_svc)
params_knn = {'kneighborsclassifier__n_neighbors': [3, 4, 5, 6, 7]}
model = make_pipeline(KNeighborsClassifier())
param_tuning(train_df, 'Survived', model, 'accuracy', params_knn)
ensembles = []
ensembles.append(('AB', AdaBoostClassifier()))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier()))
ensembles.append(('ET', ExtraTreesClassifier()))
ensembles.append(('XGB', XGBClassifier()))
results = test_models(train_df, 'Survived', ensembles , 'accuracy')
show_model_comparison_plot(results)
gb_grid_params = {'learning_rate': [0.1, 0.05, 0.02],
              'max_depth': [3, 4, 5],
              'min_samples_split': [2, 5,10] }

model = GradientBoostingClassifier()
param_tuning(train_df, 'Survived', model, 'accuracy', gb_grid_params)
model = SVC(C=1,gamma=0.1)
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

def test_model(df, features, label, model, scoring):
    num_folds = 5
    seed = 7
    X = df[features].values
    y = df[label].values
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=.30, random_state=seed)
    
    kfold = KFold(n_splits=num_folds, random_state=seed)
    result = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    return result
all_columns = train_df.drop('Survived', axis=1).columns
test_model(train_df, all_columns, 'Survived', model , 'accuracy').mean()
select = SelectPercentile(percentile=90) 
select.fit(X, y)
mask = select.get_support()
univ_features = np.array(all_columns)[mask]
print('Univariate feature selection')
print('------')
print('Excluded features')
print(np.array(all_columns)[~mask])
print('Accuracy %s' % test_model(train_df, univ_features, 'Survived', model , 'accuracy').mean())
select = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=7),
        threshold="mean")
select.fit(X, y)
mask = select.get_support()
mb_features = np.array(all_columns)[mask]
print('Model-Based Feature Selection')
print('------')
print('Excluded features')
print(np.array(all_columns)[~mask])
print('Accuracy %s' % test_model(train_df, mb_features, 'Survived', model , 'accuracy').mean())
print("Total features %s" % len(all_columns))
select = RFE(RandomForestClassifier(n_estimators=100, random_state=7),
                 n_features_to_select=18)
select.fit(X, y)
mask = select.get_support()
rfe_features = np.array(all_columns)[mask]
print('Iterative Feature Selection')
print('------')
print('Excluded features')
print(np.array(all_columns)[~mask])
print('Accuracy %s' % test_model(train_df, rfe_features, 'Survived', model , 'accuracy').mean())