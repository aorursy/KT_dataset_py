# libraries

import pandas as pd

import numpy as np

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import scale



# dataset

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# average feature values

round(train.drop('label', axis=1).mean(), 2)
# splitting into X and y

X = train.drop("label", axis = 1)

y = train['label']
X_validate = test
#X_scaled = scale(X)

#X_validate_scaled = scale(X_validate)
X_scaled = X

X_validate_scaled = X_validate
# train test split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 101)
from sklearn.decomposition import PCA

pca = PCA(whiten=True)

pca.fit(X_train)

%matplotlib inline

fig = plt.figure(figsize = (12,8))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.show()
# features required to explain 75% variance

pca = PCA(n_components= 35, whiten=True)

pca.fit(X_train)

X_train = pca.transform(X_train)

X_test = pca.transform(X_test)
# linear model



model_linear = SVC(kernel='linear')

model_linear.fit(X_train, y_train)



# predict

y_pred = model_linear.predict(X_test)

# confusion matrix and accuracy



# accuracy

print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")



# cm

print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
# non-linear model

# using rbf kernel, C=1, default value of gamma



# model

non_linear_model = SVC(kernel='rbf')



# fit

non_linear_model.fit(X_train, y_train)



# predict

y_pred = non_linear_model.predict(X_test)
# confusion matrix and accuracy



# accuracy

print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")



# cm

print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
# creating a KFold object with 5 splits 

folds = KFold(n_splits = 5, shuffle = True, random_state = 101)



# specify range of hyperparameters

# Set the parameters by cross-validation

hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]}]





# specify model

model = SVC(kernel="rbf")



# set up GridSearchCV()

model_cv = GridSearchCV(estimator = model, 

                        param_grid = hyper_params, 

                        scoring= 'accuracy', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True)
# fit the model

model_cv.fit(X_train, y_train) 
# cv results

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# converting C to numeric type for plotting on x-axis

cv_results['param_C'] = cv_results['param_C'].astype('int')



# # plotting

plt.figure(figsize=(16,6))



# subplot 1/3

plt.subplot(131)

gamma_01 = cv_results[cv_results['param_gamma']==0.01]



plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])

plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.01")

plt.ylim([0.60, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')



# subplot 2/3

plt.subplot(132)

gamma_001 = cv_results[cv_results['param_gamma']==0.001]



plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])

plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.001")

plt.ylim([0.60, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')





# subplot 3/3

plt.subplot(133)

gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]



plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])

plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.0001")

plt.ylim([0.60, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')

# printing the optimal accuracy score and hyperparameters

best_score = model_cv.best_score_

best_hyperparams = model_cv.best_params_



print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))
# model with optimal hyperparameters



# model

model = SVC(C=10, gamma=0.01, kernel="rbf")



model.fit(X_train, y_train)

y_pred = model.predict(X_test)



# metrics

print("accuracy", metrics.accuracy_score(y_test, y_pred), "\n")

print(metrics.confusion_matrix(y_test, y_pred), "\n")
# For the 'test' set

#pca.fit(X_validate_scaled)

X_validate_scaled = pca.transform(X_validate_scaled)

y_pred = model.predict(X_validate_scaled)
counter = np.arange(1,28001)

col1 = pd.DataFrame({'ImageId': counter})

col2 = pd.DataFrame({'Label': y_pred})

result = pd.concat([col1, col2], axis = 1)

result.to_csv('output_full.csv', index = False)
result.head()
# plot some of the numbers from test data to compare with predicted data

plt.figure(figsize=(14,4))

for i in range(0,30):

    plt.subplot(3,10,i+1)

    grid_data = X_validate.iloc[i].as_matrix().reshape(28,28)

    plt.imshow(grid_data, interpolation = "none", cmap = "afmhot")

    plt.xticks([])

    plt.yticks([])

plt.tight_layout()
print(y_pred[:10])

print(y_pred[10:20])

print(y_pred[20:30])