# data manipulations
import numpy as np
import pandas as pd 

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
sns.set()

# ML models
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# automated preprocessing and evaluation
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# additional
from functools import partial

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
titanic = pd.read_csv('/kaggle/input/titanic/train.csv')
titanic.shape
titanic
titanic.info()
cat_featutes = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked']
num_featutes = ['Age', 'Fare'] 
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for axi, feature in zip(ax.flat, num_featutes):
    sns.distplot(titanic[feature], ax=axi)
for feature in cat_featutes:
    x = titanic[feature].dropna()
    if np.unique(x).shape[0] < 10:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x, ax=ax)
for feature in cat_featutes:
    f = titanic[feature].dropna()
    if np.unique(f).shape[0] > 10:
        print(feature)
        print('total count:', f.shape[0])
        print('unique count:', f.unique().shape[0], end='\n\n')
catplot = partial(sns.catplot, data=titanic, height=5, aspect=1.5)
catplot(x="Pclass", y="Survived", kind="bar")
catplot(x="Embarked", y="Survived", kind="bar")
catplot(x="Pclass", y="Age", hue='Survived', kind="violin", split=True).set(ylim=0)
catplot(x="Pclass", y="Fare", hue='Survived', kind="violin", split=True).set(ylim=(0, 300))
catplot(x="Pclass", y="Survived", hue="Sex", kind="point");
useful_featues = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']  # excluding target
useful_num_featues = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']  
useful_cat_featues = ['Sex', 'Embarked']
def get_preprocessing_pipeline():
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('encode', OneHotEncoder())
    ])

    columns_pipeline = ColumnTransformer([
        ("numeric", num_pipeline, useful_num_featues),
        ("categorical", cat_pipeline, useful_cat_featues)
    ])
    final_pipeline = Pipeline([
        ('columns', columns_pipeline),
        ('pca', PCA(0.99, random_state=0))
    ])
    return final_pipeline
X = titanic.drop('Survived', axis=1)
y = titanic['Survived']
def chain_pipelines(**kwargs):
    pipelines = list(kwargs.items())
    return Pipeline(pipelines)

model = chain_pipelines(preprocessing=get_preprocessing_pipeline(),
                        model=None)
param_grid = [
    {
        'model': [LogisticRegression(n_jobs=-1)],
        'model__C': np.linspace(0.5, 5, 10)
    },
    {
        'model': [GaussianNB()]
    },
    {
        'model': [KNeighborsClassifier(n_jobs=-1)],
        'model__n_neighbors': np.arange(5, 11),
        'model__weights': ['uniform', 'distance']
    },
    {
        'model': [SVC(random_state=0)],
        'model__C': np.linspace(1, 10, 19)
    },
    {
        'model': [RandomForestClassifier(random_state=0, n_jobs=-1)],
        'model__n_estimators': [20, 25, 30, 35, 40],
        'model__max_depth': [5, 10, 15],
        'model__min_samples_leaf': [1, 2],
        'model__class_weight': ['balanced_subsample', 'balanced', None]
    }
]
grid = GridSearchCV(model, param_grid, scoring='accuracy', cv=3, n_jobs=-1)
%%time
cross_val_score(grid, X, y, cv=3, n_jobs=-1).mean()
best_ML = grid.fit(X, y).best_estimator_
best_ML.named_steps['model']
best_ML.score(X, y)
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
test_data.shape
example_sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
example_sub.head()
predictions = best_ML.predict(test_data)
test_data['Survived'] = predictions
predictions_frame = test_data[['PassengerId', 'Survived']]
predictions_frame.head()
predictions_frame.to_csv('predictions.csv', index=False)
prepr = get_preprocessing_pipeline()
X_trans = prepr.fit_transform(X)

fig, ax = plt.subplots(2, 3, 
                       figsize=(15, 10),
                       subplot_kw=dict(xticks=[], yticks=[]),
                       gridspec_kw=dict(hspace=0.01, wspace=0.01))

for axi, perplexity in zip(ax.flat, [5, 10, 15, 20, 25, 30]):
    X_2D = TSNE(perplexity=perplexity).fit_transform(X_trans)
    sns.scatterplot(X_2D[:, 0], X_2D[:, 1], hue=y, ax=axi)
fig.suptitle("t-SNE dimensionality reduction (with different perplexities)", y=0.92);