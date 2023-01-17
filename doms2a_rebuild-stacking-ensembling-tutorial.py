# Data Handling

import pandas as pd

import numpy as np



# Visualizations 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')



# Machine Learning Models / Algorithms 

import xgboost as xgb

import sklearn

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.svm import SVC

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn import metrics
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')



PassengerId = test_df['PassengerId']



print("Heads and Tail:")

display(train_df.head(5))

display(train_df.tail(5))



print("Columns Overview:")

print(train_df.columns)



print("General Description:")

display(train_df.describe())



print("Interesting Cross Tables")

display(pd.crosstab(train_df.Survived, train_df.Sex))

display(pd.crosstab(train_df.Survived, train_df.Age))

display(pd.crosstab(train_df.Survived, train_df.Pclass))

display(pd.crosstab(train_df.Survived, train_df.SibSp))

display(pd.crosstab(train_df.Survived, train_df.Parch))

display(pd.crosstab(train_df.Survived, train_df.Fare))

display(pd.crosstab(train_df.Survived, train_df.Embarked))



with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):

    print("Covariance Matrix:")

    display(train_df.cov())

    print("Correlation Matrix:")

    display(train_df.corr())
datasets = [train_df, test_df]



for dataset in datasets:

    dataset['HasCabin'] = dataset['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

    

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    

    dataset['Fare'].fillna(dataset['Fare'].mean(), inplace=True)

    dataset['CategoricalFare'] = pd.qcut(dataset['Fare'], 5, labels=False)

    

    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)

    dataset['CategoricalAge'] = pd.qcut(dataset['Age'], 5, labels=False)

    

display(datasets[0])

sex_mapping = {'male': 1, 'female': 2,}

embark_mapping = {'C': 1, 'S': 2, 'Q': 3}



for dataset in datasets:

    dataset = dataset.replace({'Sex': sex_mapping, 'Embarked': embark_mapping}, inplace=True)

# Replace last NaN values in Embarked Feautre

for dataset in datasets:

    dataset['Embarked'].fillna(dataset['Embarked'].mean(), inplace=True)
# train_df = train_df.filter(items=['Survived', 'Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked', 'HasCabin', 'FamilySize', 'IsAlone', 'CategoricalFare', 'CategoricalAge'])

# test_df = test_df.filter(items=['PassengerId', 'Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked', 'HasCabin', 'FamilySize', 'IsAlone', 'CategoricalFare', 'CategoricalAge'])

train_df = train_df.filter(items=['Survived', 'Sex', 'Age', 'Pclass', 'Fare', 'HasCabin', 'FamilySize'])

test_df = test_df.filter(items=['PassengerId', 'Sex', 'Age', 'Pclass', 'Fare', 'HasCabin', 'FamilySize'])
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train_df.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
class SklearnHelper(object):

    def __init__(self, classifier, seed=0, params=None):

        params['random_state'] = seed

        self.classifier = classifier(**params)

        

    def train(self, x_train, y_train):

        self.classifier.fit(x_train, y_train)

    

    def predict(self, x):

        return self.classifier.predict(x)

    

    def fit(self, x, y):

        return self.classifier.fit(x, y)

    

    def feature_importances(self, x, y):

        return self.classifier.fit(x, y).feature_importances_
TRAINSIZE = train_df.shape[0]

TESTSIZE = test_df.shape[0]

SEED = 0

NFOLDS = 5

kf = KFold(n_splits=NFOLDS, random_state=SEED)
def get_oof(clf, x_train, y_train, x_test):

    oof_train = np.zeros((TRAINSIZE, ))

    oof_test = np.zeros((TESTSIZE, ))

    oof_test_skf = np.empty((NFOLDS, TESTSIZE))

    

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]

        

        clf.train(x_tr, y_tr)

        

        oof_train[test_index] = clf.predict(x_te)

        

        oof_test_skf[i, :] = clf.predict(x_test)

        

    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
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
# Create 5 objects that represent our 4 models

rf = SklearnHelper(classifier=RandomForestClassifier, seed=SEED, params=rf_params)

et = SklearnHelper(classifier=ExtraTreesClassifier, seed=SEED, params=et_params)

ada = SklearnHelper(classifier=AdaBoostClassifier, seed=SEED, params=ada_params)

gb = SklearnHelper(classifier=GradientBoostingClassifier, seed=SEED, params=gb_params)

svc = SklearnHelper(classifier=SVC, seed=SEED, params=svc_params)
y_train = train_df['Survived'].ravel()

train_df = train_df.drop(['Survived'], axis=1)

test_df = test_df.drop(['PassengerId'], axis=1)

x_train = train_df.values

x_test = test_df.values
# Create our OOF train and test predictions. These base results will be used as new features

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees

rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test) # Random Forest

ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 

gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost

svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier



print("Training is complete")
cols = train_df.columns.values

# Create a dataframe with features

feature_dataframe = pd.DataFrame( {'features': cols,

    'Random Forest feature importances': rf.feature_importances(x_train,y_train),

    'Extra Trees  feature importances': et.feature_importances(x_train, y_train),

    'AdaBoost feature importances': ada.feature_importances(x_train, y_train),

    'Gradient Boost feature importances': gb.feature_importances(x_train,y_train)

})



# Create the new column containing the average of values

feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise

feature_dataframe.head(3)
y = feature_dataframe['mean'].values

x = feature_dataframe['features'].values

data = [go.Bar(

            x= x,

             y= y,

            width = 0.5,

            marker=dict(

               color = feature_dataframe['mean'].values,

            colorscale='Portland',

            showscale=True,

            reversescale = False

            ),

            opacity=0.6

        )]



layout= go.Layout(

    autosize= True,

    title= 'Barplots of Mean Feature Importance',

    hovermode= 'closest',

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='bar-direct-labels')
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),

     'ExtraTrees': et_oof_train.ravel(),

     'AdaBoost': ada_oof_train.ravel(),

      'GradientBoost': gb_oof_train.ravel()

    })

base_predictions_train.head()
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)

x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
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

 scale_pos_weight=1).fit(x_train, y_train)

predictions = gbm.predict(x_test)
# Generate Submission File 

StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId, 'Survived': predictions})

StackingSubmission.to_csv("StackingSubmission.csv", index=False)