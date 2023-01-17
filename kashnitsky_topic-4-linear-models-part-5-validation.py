import warnings

warnings.filterwarnings('ignore')



%matplotlib inline

from matplotlib import pyplot as plt

import seaborn as sns

# Graphics in retina format are more sharp and legible

%config InlineBackend.figure_format = 'retina'



import numpy as np

import pandas as pd

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier

from sklearn.model_selection import validation_curve
data = pd.read_csv('../input/telecom_churn.csv').drop('State', axis=1)

data['International plan'] = data['International plan'].map({'Yes': 1, 'No': 0})

data['Voice mail plan'] = data['Voice mail plan'].map({'Yes': 1, 'No': 0})



y = data['Churn'].astype('int').values

X = data.drop('Churn', axis=1).values
alphas = np.logspace(-2, 0, 20)

sgd_logit = SGDClassifier(loss='log', n_jobs=-1, random_state=17)

logit_pipe = Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(degree=2)), 

                       ('sgd_logit', sgd_logit)])

val_train, val_test = validation_curve(logit_pipe, X, y,

                                       'sgd_logit__alpha', alphas, cv=5,

                                        scoring='roc_auc')
def plot_with_err(x, data, **kwargs):

    mu, std = data.mean(1), data.std(1)

    lines = plt.plot(x, mu, '-', **kwargs)

    plt.fill_between(x, mu - std, mu + std, edgecolor='none',

                     facecolor=lines[0].get_color(), alpha=0.2)



plot_with_err(alphas, val_train, label='training scores')

plot_with_err(alphas, val_test, label='validation scores')

plt.xlabel(r'$\alpha$'); plt.ylabel('ROC AUC')

plt.legend();
from sklearn.model_selection import learning_curve



def plot_learning_curve(degree=2, alpha=0.01):

    train_sizes = np.linspace(0.05, 1, 20)

    logit_pipe = Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(degree=degree)), 

                           ('sgd_logit', SGDClassifier(n_jobs=-1, random_state=17, alpha=alpha))])

    N_train, val_train, val_test = learning_curve(logit_pipe,

                                                  X, y, train_sizes=train_sizes, cv=5,

                                                  scoring='roc_auc')

    plot_with_err(N_train, val_train, label='training scores')

    plot_with_err(N_train, val_test, label='validation scores')

    plt.xlabel('Training Set Size'); plt.ylabel('AUC')

    plt.legend()
plot_learning_curve(degree=2, alpha=10)
plot_learning_curve(degree=2, alpha=0.05)
plot_learning_curve(degree=2, alpha=1e-4)