%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
df = pd.read_csv('../data/fifa19.csv')
df.head()
df_sub = df[['Composure','Overall']].sample(20, random_state = 42)
df_sub.dropna(inplace = True)
df_sub.plot(x = 'Composure', y='Overall', kind = 'scatter')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

X = np.array(df_sub['Composure'])
y = df_sub['Overall']
degrees = [1,2,12]

plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                             scoring="neg_mean_squared_error", cv=10)

    X_test = np.linspace(25, 90, 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((25, 90))
    plt.ylim((60, 80))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std()))
plt.show()

df.columns
df['not_awful'] = (df.Overall > 62)*1
X = df[['not_awful','Position','Strength','Composure',
        'LongPassing','GKReflexes']]
X.dropna(inplace = True)
X_train_1 = X[X.Position != 'GK']
X_train_1.drop(['Position'],axis=1, inplace= True)
Y_train_1 = X_train_1.pop('not_awful')


X_test_1 = X[X.Position == 'GK']
X_test_1.drop(['Position'],axis=1, inplace= True)
Y_test_1 = X_test_1.pop('not_awful')
from sklearn.svm import SVC
svm = SVC(gamma ='auto')
svm.fit(X_train_1,Y_train_1)
from sklearn.metrics import accuracy_score

y_pred_train = svm.predict(X_train_1)
print(str(accuracy_score(Y_train_1, y_pred_train)))

y_pred_test = svm.predict(X_test_1)
print(str(accuracy_score(Y_test_1, y_pred_test)))

from sklearn.model_selection import train_test_split

X = df[['not_awful','Position','Strength','Composure',
        'LongPassing','GKReflexes']]
X.dropna(inplace = True)

Y = X.pop('not_awful')
X.drop('Position', axis = 1, inplace = True)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42)
svm_tt = SVC(gamma ='auto')
svm_tt.fit(X_train,y_train)
y_pred_train = svm_tt.predict(X_train)
print(str(accuracy_score(y_train, y_pred_train)))

y_pred_test = svm_tt.predict(X_test)
print(str(accuracy_score(y_test, y_pred_test)))
from sklearn.model_selection import cross_validate

svm = SVC(gamma ='auto')
cv_results = cross_validate(svm, X_train, y_train, cv = 3, n_jobs = -1)
from sklearn.model_selection import GridSearchCV 

param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']} 

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3, n_jobs = -1) 
grid.fit(X_train, y_train) 
grid.cv_results_['mean_test_score']
grid.best_score_
!pip install hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
def hyperopt_train_test(params):
    X_ = X_train[:]
    clf = SVC(**params)
    return cross_val_score(clf, X_,y_train).mean()

space4svm = {
    'C': hp.uniform('C', 7.5, 12.5),
    'kernel': hp.choice('kernel', ['rbf']),
    'gamma': hp.uniform('gamma', 0.00001, .005),
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4svm, algo=tpe.suggest, max_evals=10, trials=trials)
print('best:')
print(best)
