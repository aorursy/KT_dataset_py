import sys; print(sys.version)

import os; print(os.uname())

import sklearn; print(sklearn.__version__)



data_dir, output_dir = '../input', './'

data_dir, output_dir
for i in sorted(os.listdir(data_dir)):

    print(i)
# Python libraries

# Classic,data manipulation and linear algebra

import numpy as np

import pandas as pd

pd.set_option('display.max_columns', None)



# To draw pictures in jupyter notebook

%matplotlib inline 

import matplotlib.pyplot as plt

import seaborn as sns



# Ignore warning messages

# import warnings

# warnings.filterwarnings('ignore')
# Plots

import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.tools as tls

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)
filename = os.path.join(data_dir, 'train.csv')

train = pd.read_csv(filename)

print(train.info())

# print(train.describe())

print(train.shape)

train.head()
filename = os.path.join(data_dir, 'test.csv')

test = pd.read_csv(filename)

print(test.info())

# print(test.describe())

print(test.shape)

test.head()
def get_train_data():

    filename = os.path.join(data_dir, 'train.csv')

    train = pd.read_csv(filename, index_col=0)

    X_train = train.iloc[:, 1:]

    y_train = train['target']

    return X_train, y_train



def get_test_data():

    filename = os.path.join(data_dir, 'test.csv')

    test = pd.read_csv(filename, index_col=0)

    return test
X_train0, y_train = get_train_data()

print(X_train0.shape)



X_test0 = get_test_data()

print(X_test0.shape)



X_train0.head()
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.preprocessing import StandardScaler

from datetime import datetime



scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train0), columns=X_train0.columns, index=X_train0.index)

X_test = pd.DataFrame(scaler.transform(X_test0), columns=X_test0.columns, index=X_test0.index)



def plot_scores(c_range, scores, x_label='x'):

    plt.plot(c_range, scores)

    plt.xlabel(x_label)

    plt.ylabel('score')

    plt.show()



def scores_table(model, X, y, subtitle, n_splits=5):

    scores = ['accuracy', 'roc_auc']

    res = []

    for sc in scores:

        scores = cross_val_score(model, X, y, cv=n_splits, scoring=sc)

        res.append(scores)

    df = pd.DataFrame(res).T

    df.loc['mean'] = df.mean()

    df.loc['std'] = df.std()

    df= df.rename(columns={0: 'accuracy', 1: 'roc_auc'})



    trace = go.Table(

        header = dict(values=['<b>Fold', '<b>Accuracy','<b>Roc auc'],

                    line = dict(color='#7D7F80'),

                    fill = dict(color='#a1c3d1'),

                    align = ['center'],

                    font = dict(size = 15)),

        cells = dict(values=[(*[i+1 for i in range(n_splits)],'mean', 'std'),

                           np.round(df['accuracy'],3),

                           np.round(df['roc_auc'],3)],

                   line = dict(color='#7D7F80'),

                   fill = dict(color='#EDFAFF'),

                   align = ['center'], font = dict(size = 15)))



    layout = dict(width=800, height=400,

                  title = '<b>Cross Validation - {0} folds</b><br>{1}'.format(n_splits, subtitle),

                  font = dict(size = 15))

    fig = dict(data=[trace], layout=layout)



    py.iplot(fig, filename = 'styled_table')
# Find best hyperparameters (roc_auc)

log_clf = LogisticRegression(random_state=42)

param_grid = {

    'class_weight' : ['balanced', None], 

    'penalty' : ['l2','l1'],  

    'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000],

    'solver' : ['liblinear', 'saga'],

}

# Solver newton-cg supports only l2 penalties

# Solver lbfgs supports only l2 penalties

# Solver sag supports only l2 penalties



grid = GridSearchCV(estimator=log_clf, cv=5, param_grid=param_grid,

                    scoring='roc_auc', verbose=1, n_jobs=-1)



grid.fit(X_train, y_train)



print("Best Score: {0}".format(grid.best_score_))

print("Best Parameters: {0}".format(grid.best_params_))



best_parameters = grid.best_params_
log_clf = LogisticRegression(**best_parameters)

log_clf.fit(X_train, y_train)



selector = RFE(log_clf, 25, step=1)

selector.fit(X_train, y_train)

scores_table(selector, X_train, y_train, 'selector_clf', n_splits=10)
ranking = selector.ranking_.reshape([-1, 30])



# Plot pixel ranking

plt.matshow(ranking, cmap=plt.cm.Blues)

plt.colorbar()

plt.title("Ranking with RFE")

plt.show()
y_test = log_clf.predict_proba(X_test)
filename = os.path.join(data_dir, 'sample_submission.csv')

submission = pd.read_csv(filename)



filename = os.path.join(output_dir, 'submission.csv')

submission['target'] = y_test

submission.to_csv('submission.csv', index=False)



submission.head()