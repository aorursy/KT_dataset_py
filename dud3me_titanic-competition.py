import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
train_set = pd.read_csv("../input/titanic/train.csv")
test_set = pd.read_csv("../input/titanic/test.csv")
train_set.head()
train_set.info()
train_set.describe()
train_set["Ticket"].value_counts()
train_set['Sex'].shape
fig, ax = plt.subplots()
colors = [ "#f00" if s == 0 else "#0f0" for s in train_set["Survived"] ]
ax.scatter( train_set["Age"], train_set["Fare"], c=colors)

ax.set_xlabel("Age")
ax.set_ylabel("Fare")
plt.show()
train_set.hist(bins=50, figsize=(20,15))
# looking for correlation
corr_matrix = train_set.corr()
corr_matrix["Survived"].sort_values(ascending=False)
train_set.columns
# classification problem, use logReg
# features: ['PassengerId', 'Survived', 'Embarked', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare']
# define features to fit on
numeric_features = ['Age', 'SibSp']
categorical_features = ['Sex', 'Pclass']
# generate train sets
X_train = train_set[numeric_features + categorical_features]
y_train = train_set["Survived"]
# define the classifier to use
classifier = LogisticRegression()
classifier
# transformer for numerical attributes
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# transformer for categorical attributes
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# combine the transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])
# grid search
parameters = {'penalty': ['l2'],
              'C': [0.01, 0.1, 0.5, 1],
              'max_iter': [10, 100, 500, 1000],}

grid_search = GridSearchCV(classifier, param_grid=parameters, scoring="accuracy", cv=3, verbose=2)

grid_search.fit(preprocessor.fit_transform(X_train), (y_train))

print("-"*100)
print(grid_search.best_estimator_)
print("-"*100)
classifier = grid_search.best_estimator_
# create full prediction pipeline.
clf_pip = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', classifier)])
# fit the training data to the classifier
clf_pip.fit(X_train, y_train)
# predict on test set
X_val = test_set[numeric_features + categorical_features]
predictions = clf_pip.predict(X_val)
# generate submission file
result = pd.DataFrame({'PassengerId': test_set.PassengerId, 'Survived': predictions})
result.to_csv('submission.csv', index=False)