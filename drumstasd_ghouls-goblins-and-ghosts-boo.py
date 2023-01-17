import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

import pylab as pl



import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.info()
test_id = test['id'] # save for submission

del train['id']

del test['id']
train['type'].unique(), train['color'].unique()
sns.violinplot(x='bone_length', y='type', data=train)
sns.boxplot(x='hair_length', y='type', data=train)
sns.pairplot(train)
from category_encoders import OneHotEncoder



encoder = OneHotEncoder(cols=['color'], use_cat_names=True)



train = encoder.fit_transform(train)

test = encoder.fit_transform(test)
train.head()
from sklearn.preprocessing import LabelEncoder



encoder = LabelEncoder()



encoder.fit(train['type'])



print(encoder.classes_)



train['type_no'] = encoder.transform(train['type'])
train.head()
sns.heatmap(train.corr(), xticklabels=list(train), yticklabels=list(train))
target = train['type_no'] # for visualizations

target_string = train['type'] # for final predictions



del train['type']

del train['type_no']



target.head()
from sklearn.model_selection import train_test_split



train_data, test_data, train_target, test_target = train_test_split(train, target, test_size=0.2, random_state=42)
from mlxtend.plotting import plot_decision_regions

import matplotlib.gridspec as gridspec

import itertools



def decisions(classifier, features):

    classifier.fit(train_data[features], train_target)

    ax = plot_decision_regions(test_data[features].values, test_target.values, clf=classifier, legend=2)

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles, ['Ghost', 'Ghoul', 'Goblin'], framealpha=0.3, scatterpoints=1)

    plt.xlabel(features[0])

    plt.ylabel(features[1])

    plt.show()

    

def grid_decisions(classifiers, classifier_names, features):

    gs = gridspec.GridSpec(2, 2)

    fig = plt.figure(figsize=(10,8))



    for clf, lab, grd in zip(classifiers,classifier_names, itertools.product([0, 1], repeat=2)):

        clf.fit(train_data[features], train_target)

        ax = plt.subplot(gs[grd[0], grd[1]])

        fig = plot_decision_regions(test_data[features].values, test_target.values, clf=clf, legend=2)

        handles, labels = fig.get_legend_handles_labels()

        fig.legend(handles, ['Ghost', 'Ghoul', 'Goblin'], framealpha=0.3, scatterpoints=1)

        plt.title(lab)



    plt.show()
from sklearn.tree import DecisionTreeClassifier



decisions(DecisionTreeClassifier(), ['hair_length', 'bone_length'])
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier



clfs = [RandomForestClassifier(), AdaBoostClassifier(), SVC(), KNeighborsClassifier()]

labels = ['Random Forest', 'Ada Boost', 'Support Vector', 'K-Neighbors']



grid_decisions(clfs, labels, ['hair_length', 'bone_length'])
from IPython.display import Image

from IPython.core.display import HTML 

Image(url= 'https://cdn-images-1.medium.com/max/1600/1*JZbxrdzabrT33Yl-LrmShw.png', width=750, height=750)
train.head()
from sklearn.metrics import accuracy_score



clfs = [RandomForestClassifier(), AdaBoostClassifier(), SVC(), KNeighborsClassifier()]

labels = ['Random Forest', 'Ada Boost', 'Support Vector', 'K-Neighbors']



for model, name in zip(clfs, labels):

    model.fit(train_data, train_target)

    predictions = model.predict(test_data)

    print('{} accuracy is: {}'.format(name, accuracy_score(test_target, predictions)))
from IPython.display import Image

from IPython.core.display import HTML 

Image(url= 'https://scikit-learn.org/stable/_images/grid_search_cross_validation.png', width=500, height=500)
from sklearn.model_selection import GridSearchCV



params = {

    'C': [0.001, 0.01, 0.1, 1, 10],

    'gamma': [0.001, 0.01, 0.1, 1]

}



grid_search = GridSearchCV(SVC(), params, cv=5)



grid_search.fit(train, target)



grid_search.best_params_
import eli5

from eli5.sklearn import PermutationImportance



importance_model = SVC(C=10, gamma=0.1, probability=True)

importance_model.fit(train, target)



perm = PermutationImportance(importance_model, random_state=42).fit(test_data, test_target)

eli5.show_weights(perm, feature_names=test_data.columns.tolist())
import shap



data_for_prediction = test_data.iloc[0]



k_explainer = shap.KernelExplainer(importance_model.predict_proba, train_data)

k_shap_values = k_explainer.shap_values(data_for_prediction)



shap.initjs()

shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], data_for_prediction)
test_target.iloc[0], target_string.iloc[0]
model = SVC(C=10, gamma=0.1)

model.fit(train, target_string)
predictions = model.predict(test)

predictions[:10]
submission = pd.DataFrame({'id': test_id, 'type': predictions})

submission.head()
submission.to_csv('submission.csv', index=False)