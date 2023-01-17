

import pandas as pd

import numpy as np

import random as rnd

import re





import plotly.express as px

import plotly.graph_objects as go

import matplotlib.pyplot as plt

%matplotlib inline





from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

import itertools

import xgboost as xgb
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

combine = [train_df, test_df]

train_df.columns
import missingno as msno

msno.matrix(train_df)

msno.matrix(test_df)




def feature_plots(feature, df=train_df, labels={}):



    survived_mapping = df['Survived'].map({0: 'Dead', 1: 'Survived'})



    fig = px.histogram(df, x=survived_mapping, width=600, color=feature, labels=labels)

    fig.update_layout(

        bargap=0.2,

        xaxis_title_text='Survived',

        yaxis_title_text='Survival count'

    )

    

    return fig



feature_plots('Sex')
feature_plots('SibSp')
feature_plots('Pclass')
fig = px.histogram(train_df, x='Age', color='Survived', barmode='overlay',width= 600)

fig
fig = px.histogram(train_df, x=train_df['Age'], facet_row=train_df['Pclass'],facet_col=train_df['Survived'], width=700)



fig
for df in combine:

    

    df['Title'] = df['Name'].map(lambda x: re.search(r' ([A-Za-z]+)\.', x).group().strip().replace('.', ''))







train_df['Title'].value_counts().index
def feature_engineering(df):

    

    age_bins = [0, 5.99, 11.9, 17.9, 25.9, 47.9, 61.9, 80] #Binning Age feature based on the trend in histogram 

    age_labels = [i for i in range(1, 8)]

    title_map = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5, 'Rev': 5, 'Col': 5,

                 'Major': 5, 'Mlle': 5, 'Ms': 5, 'Countess': 5, 'Lady': 5, 'Capt': 5,

                 'Jonkheer': 5, 'Don': 5, 'Sir': 5, 'Mme': 5}

    

    

        

    df['Title'] = df['Title'].map(title_map)

    df['Title'].fillna(1,inplace=True)

    df = pd.concat([df, pd.get_dummies(df['Title'], prefix='Title')], axis=1)

    df = pd.concat([df, pd.get_dummies(df['Sex'], prefix='Sex')], axis=1)

    df = df.drop(['Ticket', 'Cabin','Name', 'PassengerId'], axis=1)

    df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))

    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    df['IsAlone'] = 0

    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'))

    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    df = pd.concat([df, pd.get_dummies(df['Pclass'], prefix='Pclass')], axis=1)

    df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)

    df = pd.concat([df, pd.get_dummies(df['AgeGroup'], prefix='AgeGroup')], axis=1)

    df = df.drop(['Age','Sex','Pclass', 'Embarked', 'Title', 'AgeGroup'], axis=1)

      

    return df

           
train_df= feature_engineering(train_df)

test_df= feature_engineering(test_df)
def scale_feature(feature):

    result = []

    

    # Applying min-max scaling to the 'Parch' and 'SipSp' features

    for df in combine:

        feature_val = df[feature]

        max_val = feature_val.max()

        min_val = feature_val.min()

        scaled_feature = (feature_val - min_val) / (max_val - min_val)

        result.append(scaled_feature)

        

    return result



train_df['SibSp'], test_df['SibSp'] = scale_feature('SibSp')

train_df['Parch'], test_df['Parch'] = scale_feature('Parch')

train_df['Fare'], test_df['Fare'] = scale_feature('Fare')

def plot_confusion_matrix(cm, classes,

                          normalize = False,

                          title = 'Confusion matrix"',

                          cmap = plt.cm.Blues) :

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation = 0)

    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :

        plt.text(j, i, cm[i, j],

                 horizontalalignment = 'center',

                 color = 'white' if cm[i, j] > thresh else 'black')



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')


train_set, test_set = train_test_split (train_df, test_size = 0.1, random_state = 42)

data = train_set

valid_data = test_set


y = valid_data.Survived.tolist()

valid_data = valid_data.drop('Survived', 1)

X = np.array(valid_data)
y = np.array(data.Survived.tolist())

data = data.drop('Survived', 1)

X = np.array(data)


skf = StratifiedKFold(n_splits=5 ,shuffle = True, random_state = 42)

for train_index, test_index in skf.split(X, y):

    X_train, y_train = X[train_index.astype(int)], y[train_index.astype(int)]

    X_test, y_test = X[test_index.astype(int)], y[test_index.astype(int)]
xgb_cfl = xgb.XGBClassifier(n_jobs = -1)



xgb_cfl.fit(X_train, y_train)

y_pred = xgb_cfl.predict(X_test)

y_score = xgb_cfl.predict_proba(X_test)[:,1]



# Confusion maxtrix & metrics

cm = confusion_matrix(y_test, y_pred)

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cm, 

                      classes=class_names, 

                      title='XGB Confusion matrix')

plt.show()

f1_score(y_test,y_pred, labels=None, pos_label=1, average= 'macro', sample_weight=None)

#show_metrics()
print(xgb_cfl.get_xgb_params())
#param_grid = {

#            'n_estimators': [700, 1000, 1200, 1300],

#            'max_depth': [3, 4, 5],

#            'min_child_weight': [1, 2]

#              }

#

#CV_xgb_cfl = GridSearchCV(estimator = xgb_cfl, param_grid = param_grid, scoring= 'f1_macro', verbose = 2)

#CV_xgb_cfl.fit(X_train, y_train)

#

#best_parameters = CV_xgb_cfl.best_params_

#print("The best parameters for using this model is", best_parameters)
xgb_cfl = xgb.XGBClassifier(n_jobs = -1, max_depth= 3,

                            n_estimators = 700, min_child_weight= 2)



xgb_cfl.fit(X_train, y_train)

y_pred = xgb_cfl.predict(X_test)

y_score = xgb_cfl.predict_proba(X_test)[:,1]



# Confusion maxtrix & metrics

cm = confusion_matrix(y_test, y_pred)

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cm, 

                      classes = class_names, 

                      title = 'XGB Confusion matrix')

plt.savefig('2.xgb_cfl_confusion_matrix.png')

plt.show()

f1_score(y_test,y_pred, labels=None, pos_label=1, average= 'macro', sample_weight=None)
test = pd.read_csv('/kaggle/input/titanic/test.csv')
PassengerId= test.PassengerId



test_df=test_df.to_numpy()
Survived=xgb_cfl.predict(test_df)
submission_df = pd.DataFrame({

    'PassengerId': test["PassengerId"],

    'Survived': Survived

})



submission_df.head()
submission_df.to_csv('submission.csv', index=False)