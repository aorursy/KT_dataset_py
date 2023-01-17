# Load general modules

import numpy as np

import pandas as pd

import statsmodels.api as sm



from sklearn.linear_model import LogisticRegression as LogReg

from sklearn.tree import DecisionTreeClassifier as DTC

from sklearn.ensemble import RandomForestClassifier as RFC

from sklearn.svm import SVC



# Load plotting modules

import plotly.offline as py

import plotly.graph_objs as go

import plotly.tools as pyt



py.init_notebook_mode()



# Load data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.shape
train.head()
train.describe()
test.shape
test.columns
train = train.drop(labels=['PassengerId', 'Name', 'Ticket'], axis=1)



# PassengerId of the testing set is needed for submissions

test_passengerid = test['PassengerId']

test = test.drop(labels=['PassengerId', 'Name', 'Ticket'], axis=1)
for col in train.columns:

    print('Null values in {}: {}'.format(col, train[col].isnull().sum()))
for col in test.columns:

    print('Null values in {}: {}'.format(col, test[col].isnull().sum()))
train['Pclass'].unique()



CabinDummy = ~ train['Cabin'].isnull()

train['Pclass'].groupby((train['Pclass'], CabinDummy)).count()
train['Cabin'] = CabinDummy.astype(int)



# the same for the testing set

test['Cabin'] = (~ test['Cabin'].isnull()).astype(int)
train['Embarked'].groupby(train['Embarked']).count()
test['Embarked'].groupby(test['Embarked']).count()
train['Embarked'] = train['Embarked'].fillna('S')

test['Embarked'] = test['Embarked'].fillna('S')
age_train = train[~train['Age'].isnull()]

age_test = train[train['Age'].isnull()]

print(age_train.shape)

print(age_test.shape)
age_train.corr()
age_regression_x_vars = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Cabin']



X_train = age_train[age_regression_x_vars].as_matrix()

y_train = age_train['Age'].as_matrix()



X_test = age_test[age_regression_x_vars].as_matrix()



lrm = sm.OLS(y_train, X_train)

lrm_results = lrm.fit()

print(lrm_results.summary())
lrm_age_predictions = lrm_results.predict(exog=X_test)
x1 = y_train

tr1 = go.Histogram(x=x1, histnorm='probability', 

                xbins=dict(start=np.min(x1), size= 5, end= np.max(x1)),

                marker=dict(color='rgb(0,0,100)'))



x2 = lrm_age_predictions

tr2 = go.Histogram(x=x2, histnorm='probability', 

                xbins=dict(start=np.min(x2), size= 5, end= np.max(x2)),

                marker=dict(color='rgb(100,0,0)'))



fig = pyt.make_subplots(rows=1, cols=2, subplot_titles=('existing age values', 'estimated age values'), shared_yaxes=True)

fig.append_trace(tr1, 1, 1)

fig.append_trace(tr2, 1, 2)



fig['layout']['xaxis1'].update(range=[0, 85])

fig['layout']['xaxis2'].update(range=[0, 85])

fig['layout'].update(height=400, width=800, title='Distribution of Age')



py.iplot(fig)
train.loc[train['Age'].isnull(), 'Age'] = lrm_age_predictions



test_missing_age = test[test['Age'].isnull()]

X_test = test_missing_age[age_regression_x_vars].as_matrix()

lrm_age_predictions_test = lrm_results.predict(exog=X_test)

test.loc[test['Age'].isnull(), 'Age'] = lrm_age_predictions_test
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
for col in train.columns:

    print('Null values in {}: {}'.format(col, train[col].isnull().sum()))
for col in test.columns:

    print('Null values in {}: {}'.format(col, test[col].isnull().sum()))
train = pd.get_dummies(train, columns=['Sex', 'Embarked'])

test = pd.get_dummies(test, columns=['Sex', 'Embarked'])



# Drop one of the dummies as otherwise we will run into multicollinearity.

train = train.drop(labels=['Sex_male', 'Embarked_Q'], axis=1)

test = test.drop(labels=['Sex_male', 'Embarked_Q'], axis=1)
train_features = train.loc[:, train.columns != 'Survived']

train_labels = train['Survived']



# Prepare a container to hold the predictions from different models

predictions = {}
log_model = LogReg(C=1)

log_model.fit(train_features, train_labels)

log_model.score(train_features, train_labels)
predictions['log_model'] = log_model.predict(test)
tree_model = DTC()

tree_model.fit(train_features, train_labels)

tree_model.score(train_features, train_labels)
predictions['tree_model'] = tree_model.predict(test)
forest_model = RFC()

forest_model.fit(train_features, train_labels)

forest_model.score(train_features, train_labels)
predictions['forest_model'] = forest_model.predict(test)
support_vector_model = SVC(C=10)

support_vector_model.fit(train_features, train_labels)

support_vector_model.score(train_features, train_labels)
predictions['support_vector_model'] = support_vector_model.predict(test)
for key in predictions:

    submission_df = pd.DataFrame({'PassengerId': test_passengerid, 'Survived': predictions[key]})

    submission_df.to_csv('submission_{}.csv'.format(key), index=False)