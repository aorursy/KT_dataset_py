import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

pd.options.display.max_rows = None

sns.set(style="whitegrid", color_codes=True)
train = pd.read_csv('../input/train.csv')
holdout = pd.read_csv('../input/test.csv')
# Read sample submission file

ss = pd.read_csv('../input/sample_submission.csv')
df=train.sample(frac=0.1,random_state=200)
df.head()
df.shape
df.info()
X = df.drop('label', axis = 1)
y = df['label']
y.shape
y.value_counts().plot(kind = 'bar')
df.describe()
# Missing values & Duplicates
train.isna().sum().sum()
test.isna().sum().sum()
train.duplicated().sum()
images = X.values.reshape(-1,28,28,1)
g = plt.imshow(images[0][:,:,0])
y.iloc[0]
# Scaling the features
X = X/255
X.describe()
from sklearn.model_selection import train_test_split
# train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
from sklearn.svm import SVC

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
base_model = SVC(kernel = 'rbf')
base_model.fit(X_train, y_train)
y_pred = base_model.predict(X_test)
print('Accuracy = {}%'.format(round(metrics.accuracy_score(y_test, y_pred),3)*100))
plt.figure(figsize = (8,5))

sns.heatmap(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred), annot = True, fmt = '0.3g')

plt.xlabel('Predicted label')

plt.ylabel('True label')
folds = KFold(n_splits = 5, shuffle = True, random_state = 101)
params = [{'gamma': [0.0001, 0.001, 0.01], 'C': [1, 10, 100, 1000]}]
model = SVC(kernel = 'rbf')
model_cv = GridSearchCV(estimator=model, param_grid = params, scoring = 'accuracy', 

                        cv = folds, verbose = 1, 

                       return_train_score=True)
model_cv.fit(X_train, y_train)
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results[['param_C', 'param_gamma','mean_train_score', 'mean_test_score']].sort_values(by = 'mean_test_score', 

                                                                                        ascending = False)
plt.figure(figsize = (16,6))





plt.subplot(1,3,1)

gamma_01 = cv_results[cv_results['param_gamma'] == 0.01]

sns.lineplot(x = 'param_C', y = 'mean_test_score', data = gamma_01)

sns.lineplot(x = 'param_C', y = 'mean_train_score', data = gamma_01)

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.ylim([0.6,1])

plt.title('Gamma = 0.01')

plt.legend(['test_accuracy', 'train_accuracy'], loc = 'upper_left')

plt.xscale('log')             



                      

plt.subplot(1,3,2)

gamma_001 = cv_results[cv_results['param_gamma'] == 0.001]

sns.lineplot(x = 'param_C', y = 'mean_test_score', data = gamma_001)

sns.lineplot(x = 'param_C', y = 'mean_train_score', data = gamma_001)

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.ylim([0.6,1])

plt.title('Gamma = 0.001')

plt.legend(['test_accuracy', 'train_accuracy'], loc = 'upper_left')

plt.xscale('log')

                      



plt.subplot(1,3,3)

gamma_0001 = cv_results[cv_results['param_gamma'] == 0.0001]

sns.lineplot(x = 'param_C', y = 'mean_test_score', data = gamma_0001)

sns.lineplot(x = 'param_C', y = 'mean_train_score', data = gamma_0001)

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.ylim([0.6,1])

plt.title('Gamma = 0.0001')

plt.legend(['test_accuracy', 'train_accuracy'], loc = 'upper_left')

plt.xscale('log')
# printing the optimal accuracy score and hyperparameters

best_score = model_cv.best_score_

best_hyperparams = model_cv.best_params_



print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))
params_2 = [{'gamma': [0.001, 0.01,0.05], 'C': [0.1,1, 10, 100]}]
model_cv_2 = GridSearchCV(estimator=model, param_grid = params_2, scoring = 'accuracy', 

                        cv = folds, verbose = 1, 

                       return_train_score=True)
model_cv_2.fit(X_train, y_train)
cv_results_2 = pd.DataFrame(model_cv_2.cv_results_)

cv_results_2[['param_C', 'param_gamma','mean_train_score', 'mean_test_score']].sort_values(by = 'mean_test_score', 

                                                                                        ascending = False)
plt.figure(figsize = (16,6))





plt.subplot(1,3,1)

gamma_05 = cv_results_2[cv_results_2['param_gamma'] == 0.05]

sns.lineplot(x = 'param_C', y = 'mean_test_score', data = gamma_05)

sns.lineplot(x = 'param_C', y = 'mean_train_score', data = gamma_05)

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.ylim([0.6,1])

plt.title('Gamma = 0.05')

plt.legend(['test_accuracy', 'train_accuracy'], loc = 'upper_left')

plt.xscale('log')             



                      

plt.subplot(1,3,2)

gamma_01 = cv_results_2[cv_results_2['param_gamma'] == 0.01]

sns.lineplot(x = 'param_C', y = 'mean_test_score', data = gamma_01)

sns.lineplot(x = 'param_C', y = 'mean_train_score', data = gamma_01)

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.ylim([0.6,1])

plt.title('Gamma = 0.01')

plt.legend(['test_accuracy', 'train_accuracy'], loc = 'upper_left')

plt.xscale('log')

                      



plt.subplot(1,3,3)

gamma_001 = cv_results_2[cv_results_2['param_gamma'] == 0.001]

sns.lineplot(x = 'param_C', y = 'mean_test_score', data = gamma_001)

sns.lineplot(x = 'param_C', y = 'mean_train_score', data = gamma_001)

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.ylim([0.6,1])

plt.title('Gamma = 0.001')

plt.legend(['test_accuracy', 'train_accuracy'], loc = 'upper_left')

plt.xscale('log')
C_final = model_cv.best_params_['C']

gamma_final = model_cv.best_params_['gamma']
model_f = SVC(C = C_final, gamma = gamma_final, kernel = 'rbf')
model_f.fit(X_train, y_train)
y_test_pred = model_f.predict(X_test)
print("Accuracy on test data = {}%".format(round(metrics.accuracy_score(y_test, y_test_pred),2)*100))
plt.figure(figsize = (8,5))

sns.heatmap(metrics.confusion_matrix(y_true=y_test, y_pred=y_test_pred), annot = True, fmt = '0.3g')

plt.xlabel('Predicted label')

plt.ylabel('True label')
holdout.head()
holdout.shape
holdout_scaled = holdout/255
holdout_pred = model_f.predict(holdout_scaled)
holdout_pred
# Checking sample submission file

ss.head()
submission = pd.DataFrame(list(zip(holdout.index, holdout_pred)), columns = ['ImageId', 'Label'])
submission['ImageId'] = submission['ImageId'].apply(lambda x: x+1)
submission.head()
submission.to_csv("Nischay_svm_mnist.csv",index=False)
submission.shape