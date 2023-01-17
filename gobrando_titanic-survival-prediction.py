# Load libraries
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

# core models for stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                             GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

PassengerId = test['PassengerId']

train.head(3)
# Any null values to worry about? 
train.isnull().sum()
full_data = [train, test]

# Encode Cabin column to track is passenger has cabin or not
for dataset in full_data:
    dataset['Has_Cabin'] = dataset['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
    
    # combine SibSp and Parch to gather FamilySize
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
    # if FamilySize = 0 mark that passenger as alone
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
    # Remove null values from Embarked
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    
# Bucket fare into categories, assign at CategoricalFare 
train['CategoricalFare'] = pd.qcut(train['Fare'], 4, labels=[1,2,3,4])

for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    # fill null age values with range of probable ages (+- 1 std) 
    age_null_random_list = np.random.randint(age_avg - age_std,
                                            age_avg + age_std, 
                                             size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
train['CategoricalAge'] = pd.cut(train['Age'], 5)

# Pull titles from passenger names and make feature
def get_title(name):
    title_search = re.search(r'([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ''

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                                'Don', 'Dr', 'Major', 'Master', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
                                               'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
# Map features
for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}) #.astype(int)
    dataset['Title'] = dataset['Title'].map({'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Rare': 4})
    dataset['Title'] = dataset['Title'].fillna(0)
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}) #.astype(int)
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'] #astype(int)
    dataset.loc[dataset['Age'] <=16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis =1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
test = test.drop(drop_elements, axis=1)
train.dtypes.sample(10)
test.isnull().sum()
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
test.isnull().sum()
# Let's make a heatmap to find correlations in features
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(), linewidths=0.1,vmax=1.0,
           square=True, cmap=colormap, linecolor='white', annot=True)
# more correlation mining with pairplots
g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex',
                       u'Parch', u'Fare', u'Embarked', 
                       u'FamilySize', u'Title']], hue='Survived',
                palette = 'seismic', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10))
g.set(xtickLabels=[])
loner_plot = sns.jointplot('IsAlone', 'Survived', data=train, kind='reg')
g = sns.violinplot(x= 'Survived', y='FamilySize', data=train, hue='Survived')
g = sns.violinplot(x= 'Survived', y='IsAlone', data=train, hue='Survived')
# build helper class for deploying sklearn classifier
# will help with ensembling multiple classifiers

# global parameters
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducing data when using random
NFOLDS = 5 # number of classifiers we're ensembling
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)
        
    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)
        
    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        return(self.clf.fit(x,y).feature_importances_)
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    
    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        
        clf.train(x_tr, y_tr)
        
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)
        
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1,1), oof_test.reshape(-1,1)
# parameters for classifiers
# Random Forrest
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'warm_start': True,
    'max_depth': 6,
    'min_samples_leaf': 2, 
    'max_features': 'sqrt',
    'verbose': 0
}

# Extra Trees
et_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'max_depth': 8, 
    'min_samples_leaf': 2, 
    'verbose': 0
}

# AdaBoost 
ada_params = {
    'n_estimators': 500,
    'learning_rate': 0.75
}

# Gradient Boosting
gb_params = {
    'n_estimators': 500,
    'max_depth': 5, 
    'min_samples_leaf': 2, 
    'verbose': 0
}

# Support Vector Classifier
svc_params = {
    'kernel':'linear',
    'C': 0.025
}
# 5 objects representing our models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
y_train = train['Survived']
# numpy array of dataset to feed our models
train = train.drop(['Survived'], axis=1)
x_train = train.values
x_test = test.values
print(len(train.values[0]))
print(x_test)
# generate base results to use as new features
rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)
gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test)
svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, x_test)
print('Training complete')
# quantify importance of features in the models
rf_features = rf.feature_importances(x_train, y_train)
et_features = et.feature_importances(x_train, y_train)
ada_features = ada.feature_importances(x_train, y_train)
gb_features = gb.feature_importances(x_train, y_train)
# test to make sure array worked
rf_features
cols = train.columns.values
feature_dataframe = pd.DataFrame( {
    'features': cols,
    'Random Forest feature importances': rf_features,
    'Extra Trees feature importances': et_features,
    'AdaBoost feature importances': ada_features,
    'Gradient Boost feature importances': gb_features
})
feature_dataframe.head()
# now we'll visualize our feature importance dataframe
# to gather intuition about what features matter

# Random Forest scatter plot 
trace = go.Scatter(
    x = feature_dataframe['features'].values,
    y = feature_dataframe['Random Forest feature importances'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['Random Forest feature importances'].values,
        colorscale='Portland',
        showscale=True),
    text = feature_dataframe['features'].values
)
data = [trace]

layout = go.Layout(
    autosize=True,
    title='Random Forest Feature Importance',
    hovermode='closest',
    yaxis=dict(
        title='Feature Importance',
        ticklen=5,
        gridwidth=2),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Extra Trees scatter plot
trace = go.Scatter(
    x = feature_dataframe['features'].values,
    y = feature_dataframe['Extra Trees feature importances'].values,
    mode = 'markers',
    marker = dict(
        sizemode = 'diameter',
        sizeref = 1, 
        size = 25, 
        color = feature_dataframe['Extra Trees feature importances'].values,
        colorscale = 'Portland',
        showscale = True),
    text = feature_dataframe['features'].values
)
data = [trace]

layout = go.Layout(
    autosize = True,
    title = 'Extra Trees Feature Importance',
    hovermode = 'closest',
    yaxis = dict(
        title = 'Feature Importance',
        ticklen = 5, 
        gridwidth = 2),
    showlegend = False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# AdaBoost scatter plot
trace = go.Scatter(
    x = feature_dataframe['features'].values,
    y = feature_dataframe['AdaBoost feature importances'].values,
    mode = 'markers',
    marker = dict(
        sizemode = 'diameter',
        sizeref = 1, 
        size = 25,
        color = feature_dataframe['AdaBoost feature importances'].values,
        colorscale = 'Portland',
        showscale = True),
    text = feature_dataframe['features'].values
)
data = [trace]

layout = go.Layout(
    autosize = True,
    title = 'AdaBoost Feature Importance',
    hovermode = 'closest',
    yaxis = dict(
        title = 'Feature Importance',
        ticklen = 5, 
        gridwidth = 2),
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter20101')

# Gradient Boost scatter plot
trace = go.Scatter(
    x = feature_dataframe['features'].values,
    y = feature_dataframe['Gradient Boost feature importances'].values,
    mode = 'markers',
    marker = dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['Gradient Boost feature importances'].values,
        colorscale= 'Portland',
        showscale = True),
    text = feature_dataframe['features'].values
)
data = [trace]

layout = go.Layout(
    autosize = True,
    title = 'Gradient Boosting Feature Importance',
    hovermode = 'closest',
    yaxis = dict(
        title = 'Feature Importance',
        ticklen = 5, 
        gridwidth = 2),
    showlegend = False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='scatter2010')
# create new column storing the average of values

feature_dataframe['mean'] = feature_dataframe.mean(axis=1)
feature_dataframe.sample(10)
# plot the average values
x = feature_dataframe['features'].values
y = feature_dataframe['mean'].values
data = [go.Bar(
            x = x,
            y = y,
            width = 0.5,
            marker = dict(
                color = feature_dataframe['mean'].values,
            colorscale = 'Portland',
            showscale = True,
            reversescale = False),
        opacity = 0.6
)]

layout = go.Layout(
    autosize = True,
    title = 'Barplots of Mean Feature Importance',
    hovermode = 'closest',
    yaxis = dict(
        title = 'Feature Importance',
        ticklen = 5,
        gridwidth = 2),
    showlegend = False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='bar-direct-labels')
# Now we'll be building a new classifier that takes in our initial predictions as a pre-train model
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
                                       'ExtraTrees': et_oof_train.ravel(),
                                       'AdaBoost': ada_oof_train.ravel(),
                                       'GradientBoost': gb_oof_train.ravel()})
base_predictions_train.head()
# correlation heatmap of second-level training
data = [
    go.Heatmap(
        x = base_predictions_train.columns.values,
        y = base_predictions_train.columns.values,
            colorscale = 'Viridis',
            showscale = True,
            reversescale = True,
        z = base_predictions_train.astype(float).corr().values
    )
]
py.iplot(data, filename='labeled-headmap')
x_train = np.concatenate((et_oof_train, rf_oof_train, 
                          ada_oof_train, gb_oof_train,
                         svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test,
                        ada_oof_test, gb_oof_test,
                        svc_oof_test), axis=1)
# build and fit to XGBoost
params = ()
gbm = xgb.XGBClassifier(n_estimators = 2000,
                        max_depth = 4,
                        min_child_weight = 2,
                        gamma = 0.9, 
                        subsample = 0.8, 
                        colsample_bytree = 0.8,
                        objective = 'binary:logistic', 
                        nthread = -1,
                        scale_pos_weight = 1).fit(x_train, y_train)
predictions = gbm.predict(x_test)
StackingSubmission = pd.DataFrame({'PassengerId': PassengerId,
                                  'Survived': predictions})
StackingSubmission.to_csv('StackingSubmission.csv', index=False)
