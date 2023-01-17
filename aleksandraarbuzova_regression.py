from sklearn import datasets, linear_model, metrics

from sklearn.model_selection import train_test_split, cross_val_score

import pandas as pd

import numpy as np

from sklearn.metrics import roc_curve, auc, precision_recall_curve

import matplotlib.pyplot as plt
wine=pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

wine.head()
target = wine['quality']

data = wine.drop('quality', axis=1)  

data.shape
plt.figure(figsize=(20, 24))

plot_number= 0

for feature_name in data.columns:

    plot_number+= 1

    plt.subplot(4, 3, plot_number)

    plt.scatter(data[feature_name],target, color = 'r')

    plt.xlabel(feature_name)

    plt.ylabel('Quality')
plt.figure(figsize=(20, 24))

hist_number= 0

for feature_name in data.columns:

    hist_number+= 1

    plt.subplot(4, 3, hist_number)

    plt.hist(data[feature_name])

    plt.xlabel(feature_name)

    
train_data, test_data, train_labels, test_labels = train_test_split(data, target,test_size = 0.3)
linear_regressor = linear_model.LinearRegression()

linear_regressor.fit(train_data, train_labels)
scorer = metrics.make_scorer(metrics.mean_absolute_error, greater_is_better = True)

linear_scoring = cross_val_score(linear_regressor, data, target, scoring = scorer, cv = 10)

print('mean: {}, std: {}'.format(linear_scoring.mean(), linear_scoring.std()))
linear_regressor.coef_