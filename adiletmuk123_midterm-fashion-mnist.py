import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#Models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
#metrics and validation
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
#matlab
import matplotlib.pyplot as plt

df = pd.read_csv('../input/fashion-mnist_train.csv') # read train data
dft = pd.read_csv('../input/fashion-mnist_test.csv') # read test data

X_train = df.drop('label', axis=1)
y_train = df['label']
X_test = dft.drop('label', axis=1)
y_test = dft['label']
data = preprocessing.scale(X_train)

print(X_train.shape)
df.head()
model = LogisticRegression( solver='sag') 

model.fit(data, y_train.values.ravel())

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
parameters = [{'C': [1, 10, 100]}]

grid_search = GridSearchCV(estimator = LogisticRegression(solver = 'sag'),
                           param_grid = parameters,
                           scoring = 'r2',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(data, y_train.values.ravel())
pd.DataFrame.from_dict(grid_search.cv_results_)
print(grid_search.best_score_)
print(grid_search.best_params_)
parameters = [{'C': [1, 10, 100]}]
grid_search = GridSearchCV(estimator = LinearSVC(),
                           param_grid = parameters,
                           cv = 5,
                           n_jobs = -1)
grid_search_svc = grid_search.fit(data, y_train.values.ravel())
model2 = LinearSVC(C =1) 

model2.fit(data, y_train.values.ravel())

y_pred2 = model2.predict(X_test)

print(accuracy_score(y_test, y_pred2))
parameters = [{'alphas': [1e-2, 1e-1, 1]}]
grid_search = GridSearchCV(estimator = RidgeClassifier(solver = 'sag'),
                           param_grid = parameters,
                           scoring = 'r2',
                           cv = 5,
                           n_jobs = -1)
grid_search_ridge = grid_search.fit(data, y_train.values.ravel())
model3 = RidgeClassifier(alpha =1, solver='sag') 

model3.fit(data, y_train.values.ravel())

y_pred3 = model3.predict(X_test)

print(accuracy_score(y_test, y_pred3))
train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(RidgeClassifier(alpha=1, solver = 'sag'), data, y_train.values.ravel(), cv=5, n_jobs=None,scoring='r2',
                       return_times=True)


train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1)
_, axis = plt.subplots(1, 1, figsize=(10, 5))
axis.set_xlabel("Training examples")
axis.set_ylabel("Score")
axis.grid()
axis.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
axis.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
axis.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
axis.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
axis.legend(loc="best")
axis.set_title("Learning curve for linear regression")
plt.show()