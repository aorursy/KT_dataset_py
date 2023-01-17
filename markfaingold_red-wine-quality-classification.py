# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import time



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
data.head()
sns.countplot(x='quality', data=data)
data['quality'] = [1 if quality > 5.5 else 0 for quality in data['quality']]

sns.countplot(x='quality', data=data)
from sklearn.model_selection import train_test_split

x = data.drop('quality',axis=1)

y = data['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y)
from sklearn.svm import SVC

linear_svm = SVC(kernel='linear')

linear_svm.fit(x_train, y_train)
from sklearn.metrics import classification_report

linear_svm_predictions = linear_svm.predict(x_test)

print(classification_report(y_test, linear_svm_predictions))
# from sklearn.decomposition import PCA

# pca = PCA(n_components=6)

# scaled_x_train = pca.fit_transform(x_train)

# scaled_x_test = pca.transform(x_test)
from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(x_train)

scaled_x_train = scaler.transform(x_train)

scaled_x_test = scaler.transform(x_test)
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV



# find best parameters for SVC using GridSearch

gamma_range = np.logspace(-3, 1, 20)

C_range = [1, 5, 10]

params = dict(gamma=gamma_range, C=C_range)

svc = GridSearchCV(SVC(), params, cv=5)

start = time.time()

svc.fit(scaled_x_train, y_train)

end = time.time()

print(end - start)
# plot SVC GridSearch results

scores = np.array(svc.cv_results_['mean_test_score']).reshape(len(C_range), len(gamma_range))

for ind, i in enumerate(C_range):

    plt.plot(gamma_range, scores[ind], label='C: ' + str(i))

plt.legend()

plt.xlabel('Gamma')

plt.ylabel('Mean Accuracy')

plt.xscale('log')

plt.show()
print(svc.best_params_)

print(svc.best_score_)
svc_predictions = svc.predict(scaled_x_test)

print(classification_report(y_test, svc_predictions))
from sklearn.ensemble import AdaBoostClassifier



# find best parameters for AdaBoost using GridSearch

n_estimators_range = np.arange(100, 1000, 100)

learning_rate_range = [0.001, 0.01, 0.1, 1]

params = dict(n_estimators=n_estimators_range, learning_rate=learning_rate_range)

abc = GridSearchCV(AdaBoostClassifier(), params, cv=5)

start = time.time()

abc.fit(scaled_x_train, y_train)

end = time.time()

print(end - start)
# plot AdaBoost GridSearch results

scores = np.array(abc.cv_results_['mean_test_score']).reshape(len(learning_rate_range), len(n_estimators_range))

for ind, i in enumerate(learning_rate_range):

    plt.plot(n_estimators_range, scores[ind], label='Learning Rate: ' + str(i))

plt.legend()

plt.xlabel('Max Estimators')

plt.ylabel('Mean Accuracy')

plt.show()
print(abc.best_params_)

print(abc.best_score_)
abc_predictions = abc.predict(scaled_x_test)

print(classification_report(y_test, abc_predictions))
from sklearn.neural_network import MLPClassifier



alpha_range = 10.0 ** -np.arange(3, 6)

hidden_layer_sizes_range = np.arange(50, 150, 5)

params = dict(alpha=alpha_range, hidden_layer_sizes=hidden_layer_sizes_range)

nn = GridSearchCV(MLPClassifier(max_iter=1000, solver='lbfgs'), params, cv=5)

start = time.time()

nn.fit(scaled_x_train, y_train)

end = time.time()

print(end - start)
# plot MLPC GridSearch results

scores = np.array(nn.cv_results_['mean_test_score']).reshape(len(alpha_range), len(hidden_layer_sizes_range))

for ind, i in enumerate(alpha_range):

    plt.plot(hidden_layer_sizes_range, scores[ind], label='Alpha: ' + str(i))

plt.legend()

plt.xlabel('Hidden Layer Size')

plt.ylabel('Mean Accuracy')

plt.show()
print(nn.best_params_)

print(nn.best_score_)
nn_predictions = nn.predict(scaled_x_test)

print(classification_report(y_test, nn_predictions))
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):

    a = 1.0 * np.array(data)

    n = len(a)

    m, se = np.mean(a), scipy.stats.sem(a)

    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

    return m, h
import numpy as np, scipy.stats as st

from sklearn.model_selection import StratifiedKFold



skf = StratifiedKFold(n_splits=40)



nn_predictions_skf = []

abc_predictions_skf = []

svc_predictions_skf = []

y_test = np.array(y_test)

for test_index, _ in skf.split(scaled_x_test, y_test):

    y = y_test[test_index]

    x = scaled_x_test[test_index]

    nn_predictions = nn.predict(x)

    f1 = classification_report(y, nn_predictions, output_dict=True)['weighted avg']['f1-score']

    nn_predictions_skf.append(f1)

    abc_predictions = abc.predict(x)

    f1 = classification_report(y, abc_predictions, output_dict=True)['weighted avg']['f1-score']

    abc_predictions_skf.append(f1)

    svc_predictions = svc.predict(x)

    f1 = classification_report(y, svc_predictions, output_dict=True)['weighted avg']['f1-score']

    svc_predictions_skf.append(f1)

print(mean_confidence_interval(nn_predictions_skf))

print(mean_confidence_interval(abc_predictions_skf))

print(mean_confidence_interval(svc_predictions_skf))