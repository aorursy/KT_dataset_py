# Load in our libraries

import pandas as pd

import numpy as np

import re

import sklearn

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')



# Going to use these 5 base models for the stacking

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.svm import SVC

from sklearn.cross_validation import KFold;

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn import preprocessing
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



# 补充embarked的缺失值

train['Embarked'] = train['Embarked'].fillna('C')



# 补充fare的缺失值

fare_median = test[(test['Pclass'] == 3) & (test['Embarked'] == 'S')]['Fare'].median()

test['Fare'] = test['Fare'].fillna(fare_median)



# 提取新的特征Deck

train['Deck'] = train['Cabin'].str[0]

test['Deck'] = test['Cabin'].str[0]

train['Deck'] = train['Deck'].fillna('Z')

test['Deck'] = test['Deck'].fillna('Z')



# 提取新的特征family type

train['Family'] = train['SibSp'] + train['Parch'] + 1

test['Family'] = test['SibSp'] + test['Parch'] + 1



train.loc[train['Family'] == 1, 'FamilyType'] = 'singleton'

train.loc[(train['Family'] > 1) & (train['Family'] < 5), 'FamilyType'] = 'small'

train.loc[train['Family'] > 4, 'FamilyType'] = 'large'



test.loc[test['Family'] == 1, 'FamilyType'] = 'singleton'

test.loc[(test['Family'] > 1) & (test['Family'] < 5), 'FamilyType'] = 'small'

test.loc[test['Family'] > 4, 'FamilyType'] = 'large'



# 提取新的特征title

def get_title(name):

    title = re.compile('(.*, )|(\\..*)').sub('',name)

    return title



titles = train['Name'].apply(get_title)

train['Title'] = titles



titles = test['Name'].apply(get_title)

test['Title'] = titles



rare_title = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']

train.loc[train["Title"] == "Mlle", "Title"] = 'Miss'

train.loc[train["Title"] == "Ms", "Title"] = 'Miss'

train.loc[train["Title"] == "Mme", "Title"] = 'Mrs'

train.loc[train["Title"] == "Dona", "Title"] = 'Rare Title'

train.loc[train["Title"] == "Lady", "Title"] = 'Rare Title'

train.loc[train["Title"] == "Countess", "Title"] = 'Rare Title'

train.loc[train["Title"] == "Capt", "Title"] = 'Rare Title'

train.loc[train["Title"] == "Col", "Title"] = 'Rare Title'

train.loc[train["Title"] == "Don", "Title"] = 'Rare Title'

train.loc[train["Title"] == "Major", "Title"] = 'Rare Title'

train.loc[train["Title"] == "Rev", "Title"] = 'Rare Title'

train.loc[train["Title"] == "Sir", "Title"] = 'Rare Title'

train.loc[train["Title"] == "Jonkheer", "Title"] = 'Rare Title'

train.loc[train["Title"] == "Dr", "Title"] = 'Rare Title'



test.loc[test["Title"] == "Mlle", "Title"] = 'Miss'

test.loc[test["Title"] == "Ms", "Title"] = 'Miss'

test.loc[test["Title"] == "Mme", "Title"] = 'Mrs'

test.loc[test["Title"] == "Dona", "Title"] = 'Rare Title'

test.loc[test["Title"] == "Lady", "Title"] = 'Rare Title'

test.loc[test["Title"] == "Countess", "Title"] = 'Rare Title'

test.loc[test["Title"] == "Capt", "Title"] = 'Rare Title'

test.loc[test["Title"] == "Col", "Title"] = 'Rare Title'

test.loc[test["Title"] == "Don", "Title"] = 'Rare Title'

test.loc[test["Title"] == "Major", "Title"] = 'Rare Title'

test.loc[test["Title"] == "Rev", "Title"] = 'Rare Title'

test.loc[test["Title"] == "Sir", "Title"] = 'Rare Title'

test.loc[test["Title"] == "Jonkheer", "Title"] = 'Rare Title'

test.loc[test["Title"] == "Dr", "Title"] = 'Rare Title'



# 转变离散数据为one-hot

labelEnc=LabelEncoder()



cat_vars=['Embarked','Sex',"Title","FamilyType",'Deck']

for col in cat_vars:

    train[col]=labelEnc.fit_transform(train[col])

    test[col]=labelEnc.fit_transform(test[col])

    

# Age缺失值填充

def fill_missing_age(data):

    

    #Feature set

    features = data[['Age','Embarked','Fare', 'Parch', 'SibSp',

                 'Title','Pclass','Family',

                 'FamilyType', 'Deck']]

    # Split sets into train and prediction

    train  = features.loc[ (data.Age.notnull()) ]# known Age values

    prediction = features.loc[ (data.Age.isnull()) ]# null Ages

    

    # All age values are stored in a target array

    y = train.values[:, 0]

    

    # All the other values are stored in the feature array

    X = train.values[:, 1::]

    

    # Create and fit a model

    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)

    rtr.fit(X, y)

    

    # Use the fitted model to predict the missing values

    predictedAges = rtr.predict(prediction.values[:, 1::])

    

    # Assign those predictions to the full data set

    data.loc[ (data.Age.isnull()), 'Age' ] = predictedAges 

    

    return data



train=fill_missing_age(train)

test=fill_missing_age(test)



# 数据归一化

std_scale = preprocessing.StandardScaler().fit(train[['Age', 'Fare']])

train[['Age', 'Fare']] = std_scale.transform(train[['Age', 'Fare']])





std_scale = preprocessing.StandardScaler().fit(test[['Age', 'Fare']])

test[['Age', 'Fare']] = std_scale.transform(test[['Age', 'Fare']])
train.info()
n_train = train.shape[0]

n_test = test.shape[0]

SEED = 0

N_FOLDS = 5

kf = KFold(n_train, n_folds=N_FOLDS, random_state=SEED)



print(n_train)

print(n_test)
class SklearnHelper(object):

    def __init__(self, classifier, seed=0, params=None):

        params['random_state'] = seed

        self.classifier = classifier(**params)

        

    def train(self, X_train, y_train):

        self.classifier.fit(X_train, y_train)

    

    def predict(self, X_test):

        return self.classifier.predict(X_test)

    

    def fit(self, X_train, y_train):

        return self.classifier.fit(X_train, y_train)

    

    def feature_importances(self, X_train, y_train):

        print(self.classifier.fit(X_train, y_train).feature_importances_)
def get_K_Fold(classifier, X_train, y_train, X_test):

    new_train = np.zeros((n_train, )) # shape = [891, 1]

    new_test = np.zeros((n_test, )) # shape = [418, 1]

    each_fold_test_records = np.empty((N_FOLDS, n_test))

    

    for i, (train_index, test_index) in enumerate(kf):

        x_tr = X_train[train_index]

        y_tr = y_train[train_index]

        x_te = X_train[test_index]

        

        # 生成新的训练集

        classifier.train(x_tr, y_tr) # 使用k-1份作为训练集

        new_train[test_index] = classifier.predict(x_te) # 1份作为测试集，预测y_te

        

        # 生成新的测试集

        each_fold_test_records[i, :] = classifier.predict(X_test)

        

    new_test[:] = each_fold_test_records.mean(axis=0)

    return new_train.reshape(-1, 1), new_test.reshape(-1 ,1)
# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 500,

     'warm_start': True, 

     #'max_features': 0.2,

    'max_depth': 6,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'verbose': 0

}



# Extra Trees Parameters

et_params = {

    'n_jobs': -1,

    'n_estimators':500,

    #'max_features': 0.5,

    'max_depth': 8,

    'min_samples_leaf': 2,

    'verbose': 0

}



# AdaBoost parameters

ada_params = {

    'n_estimators': 500,

    'learning_rate' : 0.75

}



# Gradient Boosting parameters

gb_params = {

    'n_estimators': 500,

     #'max_features': 0.2,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'verbose': 0

}



# Support Vector Classifier parameters 

svc_params = {

    'kernel' : 'linear',

    'C' : 0.025

    }
# Random Forest Model

rf = SklearnHelper(classifier=RandomForestClassifier, seed=SEED, params=rf_params)



# Extra Tree Model

et = SklearnHelper(classifier=ExtraTreesClassifier, seed=SEED, params=et_params)



# AdaBoost Model

ada = SklearnHelper(classifier=AdaBoostClassifier, seed=SEED, params=ada_params)



# Gradient Boosting Model

gb = SklearnHelper(classifier=GradientBoostingClassifier, seed=SEED, params=gb_params)



# Support Vector Classifier Model

svc = SklearnHelper(classifier=SVC, seed=SEED, params=svc_params)
features = ["Pclass", "Sex", "Age","SibSp", "Parch", "Fare",

             "Embarked", "FamilyType", "Title","Deck"]



y_train = train['Survived'].ravel()

X_train = train[features].values

X_test = test[features].values



PassengerId = test['PassengerId']

# Random Forest Model

rf_new_train, rf_new_test = get_K_Fold(rf, X_train, y_train, X_test)



# Extra Tree Model

et_new_train, et_new_test = get_K_Fold(et, X_train, y_train, X_test)



# AdaBoost Model

ada_new_train, ada_new_test = get_K_Fold(ada, X_train, y_train, X_test)



# Gradient Boosting Model

gb_new_train, gb_new_test = get_K_Fold(gb, X_train, y_train, X_test)



# Support Vector Classifier Model 

svc_new_train, svc_new_test = get_K_Fold(svc, X_train, y_train, X_test)
base_predictions_train = pd.DataFrame( {'RandomForest': rf_new_train.ravel(),

     'ExtraTrees': et_new_train.ravel(),

     'AdaBoost': ada_new_train.ravel(),

      'GradientBoost': gb_new_train.ravel()

    })



base_predictions_train.head()
new_X_train = np.concatenate(( et_new_train, rf_new_train, ada_new_train, gb_new_train, svc_new_train), axis=1)

new_X_test = np.concatenate(( et_new_test, rf_new_test, ada_new_test, gb_new_test, svc_new_test), axis=1)



print(new_X_train.shape)

print(new_X_test.shape)
gbm = xgb.XGBClassifier(

    #learning_rate = 0.02,

 n_estimators= 2000,

 max_depth= 4,

 min_child_weight= 2,

 #gamma=1,

 gamma=0.9,                        

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread= -1,

 scale_pos_weight=1).fit(new_X_train, y_train)



predictions = gbm.predict(new_X_test)
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': predictions })

StackingSubmission.to_csv("StackingSubmission.csv", index=False)