# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/voice.csv')

df.head()
print("Total number of samples: {}".format(df.shape[0]))

print("Number of male: {}".format(df[df.label == 'male'].shape[0]))

print("Number of female: {}".format(df[df.label == 'female'].shape[0]))
df.isnull().sum()
import warnings

warnings.filterwarnings("ignore")

import seaborn

df.head()

df.plot(kind='scatter', x='meanfreq', y='dfrange')

df.plot(kind='kde', y='meanfreq')

#seaborn.pairplot(df['meanfreq', 'sd', 'skew'], hue='label', size=2)
seaborn.pairplot(df[['meanfreq', 'Q25', 'Q75', 'skew', 'centroid', 'label']], 

                 hue='label', size=2)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values



# Encode label category

# male -> 1

# female -> 0



gender_encoder = LabelEncoder()

y = gender_encoder.fit_transform(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.pipeline import Pipeline



pipe_svc = Pipeline([('std_scl', StandardScaler()), 

                    ('pca', PCA(n_components=10)),

                    ('svc', SVC(random_state=1))])



pipe_svc.fit(X_train, y_train)



print('Test Accuracy: %.3f' % pipe_svc.score(X_test, y_test))
from sklearn.model_selection import cross_val_score



scores = cross_val_score(estimator=pipe_svc,

                        X=X_train,

                        y=y_train,

                        cv=10,

                        n_jobs=1)



print('Cross validation scores: %s' % scores)



import matplotlib.pyplot as plt

plt.title('Cross validation scores')

plt.scatter(np.arange(len(scores)), scores)

plt.axhline(y=np.mean(scores), color='g') # Mean value of cross validation scores

plt.show()
from sklearn.model_selection import learning_curve



train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_svc,

                                                       X=X_train,

                                                       y=y_train,

                                                       train_sizes=np.linspace(0.1, 1.0, 10),

                                                       cv=10)



# Mean value of accuracy against training data

train_mean = np.mean(train_scores, axis=1)



# Standard deviation of training accuracy per number of training samples

train_std = np.std(train_scores, axis=1)



# Same as above for test data

test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)



# Plot training accuracies 

plt.plot(train_sizes, train_mean, color='red', marker='o', label='Training Accuracy')

# Plot the variance of training accuracies

plt.fill_between(train_sizes,

                train_mean + train_std,

                train_mean - train_std,

                alpha=0.15, color='red')



# Plot for test data as training data

plt.plot(train_sizes, test_mean, color='blue', linestyle='--', marker='s', 

        label='Test Accuracy')

plt.fill_between(train_sizes,

                test_mean + test_std,

                test_mean - test_std,

                alpha=0.15, color='blue')



plt.xlabel('Number of training samples')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
from sklearn.model_selection import validation_curve



param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

train_scores, test_scores = validation_curve(estimator=pipe_svc,

                                             X=X_train,

                                             y=y_train,

                                             param_name='svc__C',

                                             param_range=param_range,

                                             cv=10)



# Mean value of accuracy against training data

train_mean = np.mean(train_scores, axis=1)



# Standard deviation of training accuracy per number of training samples

train_std = np.std(train_scores, axis=1)



# Same as above for test data

test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)



# Plot training accuracies 

plt.plot(param_range, train_mean, color='red', marker='o', label='Training Accuracy')

# Plot the variance of training accuracies

plt.fill_between(param_range,

                train_mean + train_std,

                train_mean - train_std,

                alpha=0.15, color='red')



# Plot for test data as training data

plt.plot(param_range, test_mean, color='blue', linestyle='--', marker='s', 

        label='Test Accuracy')

plt.fill_between(param_range,

                test_mean + test_std,

                test_mean - test_std,

                alpha=0.15, color='blue')



plt.xscale('log')

plt.xlabel('Regularization parameter C')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
from sklearn.model_selection import GridSearchCV



param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']},

              {'svc__C': param_range, 'svc__gamma': param_range, 

               'svc__kernel': ['rbf']}]



gs = GridSearchCV(estimator=pipe_svc,

                  param_grid=param_grid,

                  scoring='accuracy',

                  cv=10)



# Training and searching hyper parameter space and evaluating model

# by using cross validation logic folded into 10

gs = gs.fit(X_train, y_train)



print(gs.best_score_)

print(gs.best_params_)
best_model = gs.best_estimator_

best_model.fit(X_train, y_train)

print('Test Accuracy: %.3f' % best_model.score(X_test, y_test))