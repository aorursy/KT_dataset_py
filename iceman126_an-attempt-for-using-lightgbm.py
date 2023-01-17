import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
def substrings_in_string(original_string, substrings):
    for substring in substrings:
        if original_string.find(substring) != -1:
            return substring
    return np.nan

def replace_title(x):
    title = x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
    
def create_title_col(df):
    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess',
                'Don', 'Jonkheer']
    df['Title'] = df['Name'].map(lambda x: substrings_in_string(x, title_list))
    df['Title'] = df.apply(replace_title, axis=1)

def fill_nan_values(df):
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode().iloc[0], inplace=True)
    df["Fare"].fillna(df["Fare"].mean(), inplace=True)

def feature_engineering(df, fill_nan=True):
    if fill_nan is True:
        fill_nan_values(df)
    
    # Encode Suvived
    # df["Survived"] = df["Survived"].astype('category')
    
    # Encode Sex
    df.loc[df["Sex"].notna(), "Sex"] = LabelEncoder().fit_transform(df.loc[df["Sex"].notna(), "Sex"].astype('category'))
    df["Sex"] = df["Sex"].astype('category')
    
    # Encode Embarked
    df.loc[df["Embarked"].notna(), "Embarked"] = LabelEncoder().fit_transform(df.loc[df["Embarked"].notna(), "Embarked"].astype('category'))
    df["Embarked"] = df["Embarked"].astype('category')
    
    # Turn Pclass type to category
    df["Pclass"] = df["Pclass"].astype('int32')
    
    # Create Age_Band feature
    '''
    df["Age_Band"] = np.nan
    df.loc[df["Age"].notna(), "Age_Band"] = pd.cut(df.loc[df["Age"].notna(), "Age"], 5)
    df.loc[df["Age_Band"].notna(), "Age_Band"] = LabelEncoder().fit_transform(df.loc[df["Age_Band"].notna(), "Age_Band"].astype('category'))
    df["Age_Band"] = df["Age_Band"].astype('category')
    '''
    
    # Create Age_Band feature
    '''
    df["Fare_Band"] = np.nan
    df.loc[df["Fare"].notna(), "Fare_Band"] = pd.cut(df.loc[df["Fare"].notna(), "Fare"], 5)
    df.loc[df["Fare_Band"].notna(), "Fare_Band"] = LabelEncoder().fit_transform(df.loc[df["Fare_Band"].notna(), "Fare_Band"].astype('category'))
    df["Fare_Band"] = df["Fare_Band"].astype('category')
    '''
    
    # Create Family_Size feature
    df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
    df["Family_Size"] = df["Family_Size"].astype("int64")
    
    # Create Alone feature
    df["Alone"] = df["Family_Size"] > 1
    df["Alone"] = df["Alone"].astype("int").astype("category")
    
    # Create Has_Cabin feature
    df["Has_Cabin"] = 0
    df.loc[df["Cabin"].notna(), "Has_Cabin"] = 1
    df["Has_Cabin"] = df["Has_Cabin"].astype("category")
    
    # Create Pclass*Age feature
    '''
    df["Pclass*Age"] = df["Age_Band"].astype(int) * df["Pclass"].astype(int)
    df["Pclass*Age"] = df["Pclass*Age"].astype("category")
    '''
    
    # Create Title column
    create_title_col(df)
    df["Title"] = LabelEncoder().fit_transform(df["Title"])
    df["Title"] = df["Title"].astype("category")
    
    # Create Child column
    df["Child"] = df["Age"] < 12
    df["Child"] = df["Child"].astype("category")

    # Create Elder column
    df["Elder"] = df["Age"] > 65
    df["Elder"] = df["Elder"].astype("category")

    df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
train_df = pd.read_csv("../input/train.csv")
feature_engineering(train_df)

X_train = train_df.drop(["Survived"], axis=1)
y_train = train_df["Survived"]

# train_data = lgb.Dataset(X_train, label=y_train, feature_name=X_train.columns.values.tolist(), categorical_feature=["Pclass", "Sex", "Embarked"], free_raw_data=False)

clf = lgb.LGBMClassifier(objective="binary")

param_grid = {
    "num_leaves": [10],
    "n_estimators": [30],
    "max_depth":[12],
    "subsample": [0.8, 1.0],
    "subsample_freq": [2],
    "subsample_for_bin": [200],
    "reg_alpha": [0.5],
    "reg_lambda": [0.5],
    "colsample_bytree": [1.0],
}

gbm = model_selection.GridSearchCV(clf, param_grid=param_grid, cv=5, iid=False, error_score="raise", verbose=20, n_jobs=-1)
gbm.fit(X_train, y_train)

print ("{} {}".format(gbm.best_score_, gbm.best_params_))
