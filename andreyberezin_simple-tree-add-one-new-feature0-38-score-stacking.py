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
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import KFold
##from sklearn import cross_validation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold

sns.set()
df_train = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv')
print(df_train.shape)

df_test = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv')
print(df_test.shape)
df_train.head()
df_test['target'] = np.nan
df = pd.concat([df_train, df_test])
#Data analyzing
Numeric_features = [
    'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week', 'target'
]
Categorical_features = [
    'workclass', 'education', 'marital-status', 'occupation', 'relationshop',
    'race', 'sex', 'native-country'
]
sns.countplot(df['target'], label="Count")
#Check correlation
plt.figure(figsize=(12, 4))
sns.heatmap(df[Numeric_features].corr(), annot=True, cmap='Greens')


graphs = sns.FacetGrid(df, col='target')
graphs = graphs.map(sns.distplot, 'age')



df_tmp = df.loc[df['target'].notna()].groupby(['education'])['target'].agg(
    ['mean', 'std']).rename(columns={
        'mean': 'target_mean',
        'std': 'target_std'
    }).fillna(0.0).reset_index()
df = pd.merge(df, df_tmp, how='left', on=['education'])

#Feature Enginering
df.head()



df['sex'].unique()
df['race'].unique()
df['sex'] = df['sex'].replace(' Male', 0)
df['sex'] = df['sex'].replace(' Female', 1)
married = [i for i in df['marital-status'].unique() if i[:8] == ' Married']
alone = [i for i in df['marital-status'].unique() if i not in married]
df['marital-status'] = df['marital-status'].replace(married, 1)
df['marital-status'] = df['marital-status'].replace(alone, 0)
df['marital-status'].unique()
df.head()
df.drop(columns=[
    'uid', 'workclass', 'occupation','education', 'relationship', 'race', 'native-country'
],
        inplace=True)



df.head()
df.shape
test = df.loc[32561:]
test = test.drop(columns=['target'])

train = df.loc[:32560]

test.shape
train.shape
train.head()
test.head()
train['fnlwgt'] = train['fnlwgt'].astype(int)
train['education-num'] = train['education-num'].astype(int)
train['capital-gain'] = train['capital-gain'].astype(int)
train['capital-loss'] = train['capital-loss'].astype(int)
train['hours-per-week'] = train['hours-per-week'].astype(int)
train['target'] = train['target'].astype(int)
train['target_mean'] = train['target_mean'].astype(int)
train['target_std'] = train['target_std'].astype(int)

test['fnlwgt'] = test['fnlwgt'].astype(int)
test['education-num'] = test['education-num'].astype(int)
test['capital-gain'] = test['capital-gain'].astype(int)
test['capital-loss'] = test['capital-loss'].astype(int)
test['hours-per-week'] = test['hours-per-week'].astype(int)
##test['target'] = test['target'].astype(int)
test['target_mean'] = test['target_mean'].astype(int)
test['target_std'] = test['target_std'].astype(int)
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
# Some useful parameters which will come in handy later on
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 3 # set folds for out-of-fold prediction
kf = KFold( n_splits= NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier
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
        print(self.clf.fit(x,y).feature_importances_)
    
# Class to extend KNN classifer
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(y_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
# Put in our parameters for said classifiers
# Decision Tree parameters
dt_params = {
    'criterion': 'gini',
    'splitter': 'best',
    'max_features': 8,
    'max_depth': 9,
    'min_samples_split': 50,
    'min_samples_leaf': 15
}

# Decision Tree parameters 1
dt_params1 = {
    'criterion': 'gini',
    'splitter': 'best',
    'max_features': 8,
    'max_depth': 10,
    'min_samples_split': 40,
    'min_samples_leaf': 10
}

# Decision Tree parameters 2
dt_params2 = {
    'criterion': 'gini',
    'splitter': 'best',
    'max_features': 10,
    'max_depth': 10,
    'min_samples_split': 40,
    'min_samples_leaf': 15
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
dt = SklearnHelper(clf=DecisionTreeClassifier, seed=SEED, params=dt_params)
dt1 = SklearnHelper(clf=DecisionTreeClassifier, seed=SEED, params=dt_params1)
dt2 = SklearnHelper(clf=DecisionTreeClassifier, seed=SEED, params=dt_params2)
##et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
##ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
##gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
##svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)


# Create Numpy arrays of train, test and target dataframes to feed into our models
y_train = train['target'].ravel()
train = train.drop(['target'], axis=1)
x_train = train.values # Creates an array of the train data
x_test = test.values # Creats an array of the test data
# Create our OOF train and test predictions. These base results will be used as new features
dt_oof_train, dt_oof_test = get_oof(dt, x_train, y_train, x_test) # Decision Tree
dt_oof_train1, dt_oof_test1 = get_oof(dt1, x_train, y_train, x_test) # Decision Tree 1
dt_oof_train2, dt_oof_test2 = get_oof(dt2, x_train, y_train, x_test) # Decision Tree 2
##rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
##ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
##gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
##svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

print("Training is complete")
dt_feature = dt.feature_importances(x_train,y_train)
dt_feature1 = dt1.feature_importances(x_train,y_train)
dt_feature2 = dt2.feature_importances(x_train,y_train)
dt_features = [0.04750481, 0.00635014, 0.20461296, 0.40069987, 0.00573966, 0.23589245,
 0.06582338, 0.03337672, 0.    ,     0.        ]
dt_features1 = [0.04804463, 0.01061447, 0.23557017, 0.39341705, 0.00727647, 0.20161986,
 0.06373847, 0.03971887, 0.      ,   0.        ]
dt_features2 = [0.04787151, 0.01446382, 0.23620588, 0.39310963, 0.00702615, 0.19962519,
 0.06340531, 0.03829252, 0.        , 0.        ]
cols = train.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'Decision Tree feature importances': dt_features,
     'Decision Tree feature importances1': dt_features1,
     'Decision Tree feature importances2': dt_features2,                                   
    })
# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Decision Tree feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Decision Tree feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Decision Tree feature importances',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')
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
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='bar-direct-labels')
base_predictions_train = pd.DataFrame( {'Decision Tree': dt_oof_train.ravel(),
     'Decision Tree 1': dt_oof_train1.ravel(),
     'Decision Tree 2': dt_oof_train2.ravel(),
    })
base_predictions_train.head()
data = [
    go.Heatmap(
        z= base_predictions_train.astype(float).corr().values ,
        x=base_predictions_train.columns.values,
        y= base_predictions_train.columns.values,
          colorscale='Viridis',
            showscale=True,
            reversescale = True
    )
]
py.iplot(data, filename='labelled-heatmap')
x_train = np.concatenate(( dt_oof_train, dt_oof_train1, dt_oof_train2), axis=1)
x_test = np.concatenate(( dt_oof_test, dt_oof_test1, dt_oof_test2), axis=1)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(
    n_neighbors=100,
    weights='distance',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    n_jobs=-1
).fit(x_train, y_train)
predictions = knn.predict(x_test)
df_train = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv')
print(df_train.shape)

df_test = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv')
print(df_test.shape)
df_test['target'] = np.nan
df = pd.concat([df_train, df_test])
# Generate Submission File 
df_submit = pd.DataFrame({
    'uid': df.loc[df['target'].isna()]['uid'],
    'target': predictions
})
df_submit.to_csv('submitStacking.csv', index=False)
print(df_submit.shape)

