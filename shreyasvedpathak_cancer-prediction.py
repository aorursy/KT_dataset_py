import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt 
breast_data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

breast_data.head()
breast_data = breast_data.drop(['id', 'Unnamed: 32'], axis  = 1)

breast_data.head()
breast_data.describe()
breast_data['diagnosis'] = np.where((breast_data['diagnosis'] == 'M'), 1, 0)

breast_data.head()
breast_data['diagnosis'].unique()
cancer_M = breast_data[breast_data['diagnosis']==1]

cancer_B = breast_data[breast_data['diagnosis']==0]
features_mean=list(breast_data.columns[1:11])

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10,13))

axes = axes.ravel()

for idx,ax in enumerate(axes):

    ax.figure

    ax.hist([cancer_M[features_mean[idx]],cancer_B[features_mean[idx]]], bins = 50, alpha=0.8, stacked=True, label=['M','B'], color=['red','blue'])

    ax.legend(loc='upper right')

    ax.set_title(features_mean[idx])

plt.tight_layout()

plt.show()
from sklearn.model_selection import train_test_split, KFold

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
train, test = train_test_split(breast_data, test_size = 0.2, random_state = 0)
def classification_model(model, train, test, features, target):

    model.fit(train[features], np.ravel(train[target]))

    pred = model.predict(test[features])

  

    accuracy = accuracy_score(pred ,test[target])

    print("Accuracy : %s" % "{0:.3%}".format(accuracy))
features = ['radius_mean']

target = ['diagnosis']



model_logistic = LogisticRegression()

classification_model(model_logistic, train, test, features, target)
features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean','concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

target = ['diagnosis']



model_logistic_2 = LogisticRegression()

classification_model(model_logistic_2 , train, test, features, target)
features = ['radius_mean', 'perimeter_mean', 'area_mean', 'compactness_mean','concave points_mean', 'concavity_mean']

target = ['diagnosis']



model_random = RandomForestClassifier(random_state=0)

classification_model(model_random, train, test, features, target)
features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',

                 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']



model_random_2 = RandomForestClassifier(random_state=0)

classification_model(model_random_2, train, test, features, target)
imp_features = pd.Series(model_random_2.feature_importances_, index=features).sort_values(ascending=False)

imp_features
features = imp_features.index[:7]

target = ['diagnosis']



model_random_3 = RandomForestClassifier(random_state=0)

classification_model(model_random_3, train, test, features, target)