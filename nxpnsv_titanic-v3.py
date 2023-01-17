# This Python 3
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing, metrics 
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Create full data set for convenient transformation
# (drop Survived and PassengerId to reduce possible bias)
full = pd.concat([train.drop('Survived', axis=1), test], axis=0)
full.drop('PassengerId', axis=1, inplace=True)
# There are null entries!
full.isnull().sum()
# Impute Embarked and Fare with the mode and mean
full['Embarked'].fillna(full['Embarked'].mode(), inplace=True)
full['Fare'].fillna(full['Fare'].mean(), inplace = True)
# Number of nulls correlates to survivial
# Instead of imputing we can use this
def null_count(df):
    return df[["Cabin", "Age"]].apply(lambda x: x.isnull().astype(int)).sum(axis=1)
train["nnull"] = null_count(train)
print(train.groupby("nnull").agg(({'PassengerId':'size', 'Survived':'mean'})))
full["nnull"] = null_count(full) # Apply to full dataset
# Cabin type (first letter in cabin) also correlates to survival
def cabin_type(df):
    cab = df['Cabin'].astype(str).str[0] # this captures the letter
    return cab.map(
        {k: i for i, k in enumerate(cab.unique())})
train["Cabin_type"] = cabin_type(train)
# this transforms the letters into numbers
print(train.groupby("Cabin_type").agg(({'PassengerId':'size', 'Survived':'mean'})))
full["Cabin_type"] = cabin_type(train)
# We can drop no longer used columns
full.drop(["Cabin", "Age"], inplace=True, axis=1) # Drop replaced column
# Now there are no more null
full.isnull().sum()
# Titles are correlated to survival, but there are many types so we collapse titles to fewer categories
def extract_titles(df):
    titles = {
        "Mr" :         "Mr",
        "Mme":         "Mrs",
        "Ms":          "Mrs",
        "Mrs" :        "Mrs",
        "Master" :     "Master",
        "Mlle":        "Miss",
        "Miss" :       "Miss",
        "Capt":        "Officer",
        "Col":         "Officer",
        "Major":       "Officer",
        "Dr":          "Officer",
        "Rev":         "Officer",
        "Jonkheer":    "Royalty",
        "Don":         "Royalty",
        "Sir" :        "Royalty",
        "Countess":    "Royalty",
        "Dona":        "Royalty",
        "Lady" :       "Royalty"
    }
    return df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False).map(titles)
train["title"] = extract_titles(train)
# this transforms the letters into numbers
print(train.groupby("title")[["Survived"]].mean())
full["title"] = extract_titles(full)
# Make a famliy size from parch and sibsp
full["Family_size"] = full[["Parch", "SibSp"]].sum(axis=1) + 1
full.drop(["Parch", "SibSp", 'Name', 'Ticket'], inplace=True, axis=1) # Drop useless columns
# Encode sex as 0 or 1
lable_encoder = preprocessing.LabelEncoder()
lable_encoder.fit(full["Sex"])
full["Sex"] = lable_encoder.transform(full["Sex"])
# Expand categoricals to dummy booleans
dummies = pd.get_dummies(full, columns = ["title", 'nnull', 'Cabin_type', 'Embarked'])
display(dummies.head())
X = dummies[:len(train)]
new_X = dummies[len(train):]
y = train.Survived
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = .3, random_state = 1, stratify = y)
def grid_search(clf, grid, X, y, cv=10):
    gs = GridSearchCV(
        clf,
        grid,
        scoring='roc_auc',
        iid=False,
        verbose=1,
        cv=cv)
    gs.fit(X, y)
    print("Params", gs.best_params_)
    print("Score", gs.best_score_)
    return gs
xgbclf = XGBClassifier(
    learning_rate =0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    scale_pos_weight=1,
    seed=1)
# Find max_depth and min_child_weight
gs_1 = grid_search(
    xgbclf,
    {
        'max_depth':range(3,10,1),
        'min_child_weight':range(1,6,1)
    },
    X_train, y_train)
# Now find best gamma
gs_2 = grid_search(
    gs_1.best_estimator_,
    {'gamma':[i*0.1 for i in range(0,5)]},
    X_train, y_train)
# Find subsample and colsample
gs_3 = grid_search(
    gs_2.best_estimator_,
    {
         'subsample':[i*0.1 for i in range(6,10)],
         'colsample_bytree':[i*0.1 for i in range(6,10)]
    },
    X_train, y_train)
# Find regularization parameter
gs_4 = grid_search(
    gs_3.best_estimator_,
    {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]},
    X_train, y_train)
params = gs_4.best_params_
params["learning_rate"] = 0.01
xgbclf = XGBClassifier(**params)
xgbclf.fit(X_train, y_train)
xgb_pred = xgbclf.predict(new_X)
submission = pd.concat([test.PassengerId, pd.DataFrame(xgb_pred)], axis = 'columns')
submission.columns = ["PassengerId", "Survived"]
submission.to_csv('titanic_submission.csv', header = True, index = False)