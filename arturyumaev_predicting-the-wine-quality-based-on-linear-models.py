%matplotlib inline

from matplotlib import pyplot as plt

import pandas as pd

import numpy as np

import sklearn
ds = pd.read_csv("../input/winequality-red.csv", sep=",", encoding="utf-8")
ds.head(5)
from pandas.tools.plotting import scatter_matrix

scatter_matrix(ds, alpha=0.05, figsize=(18, 18));
# What we want to predict?

predict_label = 'quality'

labels = ds[predict_label]
features_labels = [l for l in ds if l != predict_label]

# Remove the last column, that we will take like the labels column
features_labels
features = ds[features_labels]
features.head(10)
labels.head(10)
import sklearn.cross_validation

features_train, features_test, labels_train, labels_test = sklearn.cross_validation.train_test_split(features, labels, test_size=80)
if ((len(features_train) == len(labels_train)) and (len(features_test) == len(labels_test))):

    print("Splitting was correct.")
alpha_start = -1

alpha = [-1e1,

        -1e0,

        -1e-1,

        -1e-2,

        -1e-3,

        -1e-4,

        1e-4,

        1e-3,

        1e-2,

        1e-1,

        1e0,

        1e1]

#step = 1e-4

#curr = alpha_start

#while curr <= abs(alpha_start):

#    alpha.append(curr)

#    curr += step
import sklearn.linear_model



model = sklearn.linear_model.Ridge(alpha=0.1)
model.fit(X=features_train, y=labels_train)
model.get_params()
y_test = model.predict(features_test)
sklearn.metrics.mean_squared_error(labels_test, y_test)
err_by_alpha_depend = []



for a in alpha:

    model = sklearn.linear_model.Ridge(a)

    model.fit(X=features_train, y=labels_train)

    y_test = model.predict(features_test)

    err_by_alpha_depend.append(sklearn.metrics.mean_squared_error(labels_test, y_test))
plt.title("Ridge")

plt.xlabel("Alpha")

plt.ylabel("MSE")

plt.plot(alpha, err_by_alpha_depend, "r");
alpha
min_err = min(err_by_alpha_depend)
index_min_err = err_by_alpha_depend.index(min_err)
print("Best MSE = {:.4f} with Alpha = {:.7f}".format(min_err, alpha[index_min_err]))
import sklearn
from sklearn.linear_model import Lasso
model2 = Lasso()
err_by_alpha_depend2 = []



for a in alpha:

    model2 = Lasso(alpha=a)

    model2.fit(X=features_train, y=labels_train)

    y_test2 = model2.predict(features_test)

    err_by_alpha_depend2.append(sklearn.metrics.mean_squared_error(labels_test, y_test2));
ds.columns[7]
from sklearn.linear_model import Ridge

model2 = Ridge(alpha=0.01)

model2.fit(X=features_train, y=labels_train)

#pd.Series(model2.coef_).plot()



import pickle

f = open('model.bin', 'wb')

pickle.dump(model2, f)

f.close()
model2.coef_
plt.title("Lasso")

plt.xlabel("Alpha")

plt.ylabel("MSE")

plt.plot(alpha, err_by_alpha_depend2, "r");
min_err2 = min(err_by_alpha_depend2)

index_min_err2 = err_by_alpha_depend2.index(min_err2)

print("Best MSE = {:.4f} with Alpha = {:.7f}".format(min_err2, alpha[index_min_err2]))
err_by_alpha_depend3 = []



positive_alpha = [a for a in alpha if (a >= 0)] # Accomplishment for this method



for a in positive_alpha:

    model3 = sklearn.linear_model.SGDRegressor(alpha=a, penalty="l1", tol=1e-3)

    model3.fit(X=features_train, y=labels_train)

    y_test3 = model3.predict(features_test)

    err_by_alpha_depend3.append(sklearn.metrics.mean_squared_error(labels_test, y_test3))



        
len(positive_alpha)
len(err_by_alpha_depend3)
plt.title("Stoch Gradient Descent Regression")

plt.xlabel("Alpha")

plt.ylabel("MSE")

plt.plot(positive_alpha, err_by_alpha_depend3, "r")
min_err3 = min(err_by_alpha_depend3)

index_min_err3 = err_by_alpha_depend3.index(min_err3)

print("Best MSE = {:.4f} with Alpha = {:.7f}".format(min_err3, alpha[index_min_err3]))
err_by_alpha_depend3

