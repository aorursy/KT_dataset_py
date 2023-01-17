import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline
titanic = pd.read_csv('/kaggle/input/titanic/train.csv')
titanic.head(10)
df.info()
df.describe()
df.isna().mean().sort_values(ascending=False)
sns.set_style("darkgrid")
titanic.hist(bins=20, figsize=(20,15))
plt.show()
cols = ['Sex', 'Pclass', 'Embarked', 'Survived']
for col in cols:
    sns.countplot(df[col])
    plt.show()
X_train = titanic.drop('Survived', axis=1)
y_train = titanic['Survived']
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot', OneHotEncoder()),
])

num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
cat_cols = ['Pclass', 'Sex', 'Embarked']

pipeline = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

X_train_tr = pipeline.fit_transform(X_train)
from sklearn.model_selection import cross_val_score

def score(estimator):
    scores = cross_val_score(estimator, X_train_tr, y_train,
                             scoring='accuracy', cv=10)
    return scores
def display_scores(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Std:', scores.std())
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(random_state=42)
rf_scores = score(rf_clf)
display_scores(rf_scores)
from sklearn.svm import SVC

svc = SVC()
svc_scores = score() 