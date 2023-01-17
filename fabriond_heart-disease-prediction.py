import numpy as np

import pandas as pd



dataset = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')



#This dataset has 0 as sick by default, so we'll just swap the target values to make them more intuitive

target_0 = dataset['target'] == 0

target_1 = dataset['target'] == 1



dataset.loc[target_0, 'target'] = 1

dataset.loc[target_1, 'target'] = 0



#According to the guide at https://www.kaggle.com/ronitf/heart-disease-uci/discussion/105877, thal being 0 or ca being 4 represent NaNs

dataset = dataset[(dataset['thal'] != 0) & (dataset['ca'] != 4)]



dataset
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']



dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])

dataset
import matplotlib.pyplot as plt

import matplotlib as mpl



mpl.rcParams['figure.figsize'] = 20, 14

plt.matshow(dataset.corr()**2, cmap= 'Blues_r')

plt.yticks(np.arange(dataset.shape[1]), dataset.columns)

plt.xticks(np.arange(dataset.shape[1]), dataset.columns)

plt.colorbar()

plt.show()
dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

dataset
X = dataset.drop('target', 'columns').values

y = dataset['target'].values



X.shape
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



selector = SelectKBest(chi2, k='all')

selector.fit(X,y)



pd.DataFrame(np.transpose([selector.scores_.round(4), selector.pvalues_.round(4)]), dataset.drop('target', 'columns').columns, ['Chi-Square', 'P-Values'])
import matplotlib.pyplot as plt



mpl.rcParams['figure.figsize'] = 10, 14



plt.subplot(2, 1, 1)

plt.bar(np.arange(len(selector.scores_)), selector.scores_)

plt.xticks(np.arange(len(selector.scores_)), dataset.drop('target', 'columns').columns, rotation=90)

plt.yticks(np.arange(0, 50, step=5))

plt.ylabel('Chi-Square')

plt.grid(True, axis='y')



plt.subplot(2, 1, 2)

plt.bar(np.arange(len(selector.pvalues_)), selector.pvalues_)

plt.xticks(np.arange(len(selector.pvalues_)), dataset.drop('target', 'columns').columns, rotation=90)

plt.yticks(np.arange(0.05, 1.0, step=0.1))

plt.ylabel('P-Values')

plt.grid(True, axis='y')



plt.show()
from sklearn.model_selection import cross_validate



scores = {}



metrics = ['balanced_accuracy', 'f1_weighted', 'neg_log_loss']
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100, criterion='gini')

scores["Random Forest with all features"] = cross_validate(random_forest, X, y, scoring=metrics, cv=10, return_estimator=True)
selector = SelectKBest(chi2, k=(selector.pvalues_ > 0.05).sum())

X = selector.fit_transform(X,y)

scores["Random Forest with best features only"] = cross_validate(random_forest, X, y, scoring=metrics, cv=10, return_estimator=True)
flag = False

for algorithm in scores:

    if flag : print()

    flag = True

    print(algorithm+':')

    for metric in metrics:

        metric_name = ' '.join([s.capitalize() for s in metric.split('_')])

        print('\tAverage', metric_name+':', scores[algorithm]['test_'+metric].mean())