import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv(r'../input/data.csv')

print(df.shape)

print(df.dtypes)
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})
X, y = df.iloc[:,1:31]   ,df.iloc[:,0]
print(X.shape, y.shape)
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=7, test_size=0.2)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

pipe_lr =  Pipeline([('scl', StandardScaler()),

                    ('clf', LogisticRegression(random_state=1))])

pipe_lr.fit(X_train, y_train)

pipe_lr.score(X_test, y_test)
from sklearn.model_selection import cross_val_score

score = cross_val_score(pipe_lr, 

                       X_train, y_train, cv = 10)

print('cv mean accuracy with a +/- std ',(np.mean(score), np.std(score)))
from sklearn.model_selection import learning_curve

pipe_lr = Pipeline([('scl', StandardScaler()),

                   ('clf', LogisticRegression(random_state=0))

                   ])

train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr,

                                                       X=X_train, y=y_train,

                                                       cv=10, train_sizes = np.linspace(0.1,1,10))

train_mean = np.mean(train_scores, axis=1)

train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)

test_std = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_mean, 'o-', color='red', label='Training_score')

plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.15, color='red')

plt.plot(train_sizes, test_mean, 'o-', color='black', label='Validation_score')

plt.fill_between(train_sizes, test_mean-test_std, test_mean+test_std, alpha=0.15, color='black')

plt.legend(loc='lower right')

plt.xlabel('number of training samples')

plt.ylabel('accuracy')

plt.ylim([0.8, 1.0])

plt.show()
# from validation curve the best value comes out for C=0.1 

from sklearn.model_selection import validation_curve

param_range = [0.0001, 0.001,0.01,0.1,1.0,10.0,100.0]

train_scores, test_scores = validation_curve(estimator=pipe_lr,

                                            X=X_train, y=y_train,

                                            cv=10,

                                            param_range=param_range,

                                            param_name='clf__C')

train_mean = np.mean(train_scores, axis=1)

train_std = np.mean(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)

test_std = np.mean(test_scores, axis=1)

plt.plot(param_range, train_mean, 'o-', color='red', label='training accuracy')

plt.fill_between(param_range, train_mean-train_std, train_mean+train_std, color='red', alpha=0.1)

plt.plot(param_range, test_mean, 'o-', color='black', label='validation_accuracy')

plt.fill_between(param_range, test_mean-test_std, test_mean+test_std, color='black', alpha=0.1)

plt.xlabel('parameter C')

plt.ylabel('accuracy')

plt.legend(loc='lower left')

plt.ylim([0.8,1.0])

plt.xscale('log')

plt.show()
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

pipe_svc = Pipeline([('scl', StandardScaler()),

                    ('svc', SVC(random_state=1))

                    ])

param_range = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]

param_grid = [{'svc__C': param_range,

              'svc__kernel':['linear']},

             {'svc__C':param_range,

             'svc__kernel':['rbf'],

             'svc__gamma':param_range}]

grid_search = GridSearchCV(estimator=pipe_svc,

                          param_grid=param_grid,

                          scoring='accuracy',

                          cv=10)

grid_search.fit(X_train, y_train)

print(grid_search.best_score_)

print(grid_search.best_params_)
from sklearn.metrics import confusion_matrix

pipe_svc.fit(X_train, y_train)

y_pred_svc = pipe_svc.predict(X_test)

y_pred_lr = pipe_lr.predict(X_test)

mat_svc = confusion_matrix(y_true=y_test, y_pred=y_pred_svc)

mat_lr = confusion_matrix(y_true=y_test, y_pred=y_pred_lr)

print(mat_svc)

print(mat_lr)
plt.figure(figsize=(5,5))

plt.matshow(mat_svc, alpha=0.6)

for i in range(len(mat_svc[0])):

    for j in range(len(mat_svc[1])):

        plt.text(x=j, y=i, s=mat_svc[i,j],

                va='center', ha='center')

plt.xlabel('predicted class')

plt.ylabel('actual class')

plt.title('confusion_matrix using svc()')

plt.matshow(mat_lr, alpha=0.6)

for k in range(len(mat_lr[0])):

    for l in range(len(mat_lr[1])):

        plt.text(x=l, y=k, s=mat_lr[k,l],

                va='center', ha='center')

plt.xlabel('predicted class')

plt.ylabel('actual class')

plt.title('confusion_matrix using lr()')

plt.show()