import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')



from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.compose import ColumnTransformer

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier



from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import cross_val_score, ShuffleSplit
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
train_df = pd.read_csv("../input/titanic/train.csv")

test_df = pd.read_csv("../input/titanic/test.csv")
train_df.head()
categorical_cols = train_df.dtypes[train_df.dtypes == 'object'].index

numerical_cols = train_df.dtypes[train_df.dtypes != 'object'].index



print("Categorical Columns : \n", "\n".join([str(col) for col in categorical_cols]))

print()

print("Numerical Columns : \n", "\n".join([str(col) for col in numerical_cols]))
sns.countplot('Sex', data=train_df)
plt.figure(figsize=(12, 8))

grid = sns.FacetGrid(train_df, col='Survived', row='Sex', aspect=1.6)

grid.map(sns.distplot, 'Age', kde=False, bins=20)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', aspect=1.6)

grid.map(sns.countplot, 'Sex')
grid = sns.FacetGrid(train_df, col='Survived', aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare')
grid = sns.FacetGrid(temp_df, col='Survived', row='Embarked', aspect=1.6)

grid.map(sns.countplot, 'Sex')
sns.distplot(train_df[(train_df['Survived'] == 1) & (train_df['Sex'] == 'female')]['Fare'], label='Females', kde=False)

sns.distplot(train_df[(train_df['Survived'] == 1) & (train_df['Sex'] == 'male')]['Fare'], label='Males', kde=False)

plt.legend()
sns.countplot('Embarked', data=train_df)
for i, dataset in enumerate([train_df, test_df]):

    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())

    dataset['Cabin'] = dataset['Cabin'].fillna(dataset['Cabin'].mode()[0])

    dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])

    

    if i == 0:

        train_df = dataset

    else:

        test_df = dataset



test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].mean())
test_df.info()
for i, dataset in enumerate([train_df, test_df]):    

    dataset['FamilySize'] = dataset['Parch'] + dataset['SibSp']

    dataset['Embarked'] = dataset['Embarked'].map({'S': 1, 'C': 2, 'Q': 3}).astype(int)

    dataset = dataset.drop(['Ticket', 'Cabin', 'Name', 'Parch', 'SibSp'], axis=1)

    if i == 0:

        train_df = dataset

    else:

        test_df = dataset

train_df.info()
ids = test_df['PassengerId']

train_df = train_df.drop(['PassengerId'], axis=1)

test_df = test_df.drop(['PassengerId'], axis=1)



train_df['Pclass'] = train_df['Pclass'].astype(str)

train_df['Embarked'] = train_df['Embarked'].astype(str)

test_df['Pclass'] = test_df['Pclass'].astype(str)

test_df['Embarked'] = test_df['Embarked'].astype(str)



# this will turn all the categorical variables into one hot encoding

train_df = pd.get_dummies(train_df)

test_df = pd.get_dummies(test_df)
X_train = train_df.iloc[:, 1:].values

y_train = train_df['Survived'].values

X_test = test_df[:].values



ct = ColumnTransformer([('standard_scaling', StandardScaler(), [0, 1])], remainder='passthrough')

X_train = ct.fit_transform(X_train)

X_test = ct.transform(X_test)

X_train.shape, y_train.shape, X_test.shape
X_train[0], X_test[0]
# helper function

def print_stats(model):

    y_pred = model.predict(X_train)

    cm = confusion_matrix(y_train, y_pred)



    print("Confusion Matrix : \n", cm)

    print("Correct Predictions  : ", cm[0, 0] + cm[1, 1])

    print("Incorrect Predictions : ", cm[0, 1] + cm[1, 0])

    print("{} Score : {:.2f}%".format(model.__class__.__name__, model.score(X_train, y_train) * 100))
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

print_stats(log_reg)
svm = SVC()

svm.fit(X_train, y_train)

print_stats(svm)
grad_boost = GradientBoostingClassifier()

grad_boost.fit(X_train, y_train)

print_stats(grad_boost)
rand_for = RandomForestClassifier(n_estimators=100)

rand_for.fit(X_train, y_train)

print_stats(rand_for)
desc_tree = DecisionTreeClassifier()

desc_tree.fit(X_train, y_train)

print_stats(desc_tree)
ada_boost = AdaBoostClassifier()

ada_boost.fit(X_train, y_train)

print_stats(ada_boost)
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

print_stats(knn)
from xgboost import XGBClassifier, XGBRFClassifier



xgb = XGBClassifier()

xgb.fit(X_train, y_train)

print_stats(xgb)
xgbrf = XGBRFClassifier()

xgbrf.fit(X_train, y_train)

print_stats(xgbrf)
# cv = ShuffleSplit(n_splits=5, test_size=0.2)

# rand_for_cv = cross_val_score(rand_for, X_train, y_train, cv=cv) # Random Forest Classifier

# desc_tree_cv = cross_val_score(desc_tree, X_train, y_train, cv=cv) # Decision Tree Classifier

# xgb_cv = cross_val_score(xgb, X_train, y_train, cv=cv) # XGBoost Classifier

print("Random Forest Mean CV Score : {:.2f}".format(np.mean(rand_for_cv)))

print("Decision Tree Mean CV Score : {:.2f}".format(np.mean(desc_tree_cv)))

print("XGBoost Mean CV Score : {:.2f}".format(np.mean(xgb_cv)))
test_df.head(), X_test[0]
predictions = rand_for.predict(X_test)

output = pd.DataFrame({"PassengerId": ids, "Survived": predictions})

output.head(20)
output.to_csv('submission.csv', index=False)
SUBMISSION = True