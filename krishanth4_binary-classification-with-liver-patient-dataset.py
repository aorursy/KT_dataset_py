# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/indian_liver_patient.csv')
data.info()
data.describe()
data.Albumin_and_Globulin_Ratio.fillna(data.Albumin_and_Globulin_Ratio.mean(), inplace=True)
data.Dataset.value_counts()
#Assuming 1=no disease and 2=disease, mapping the below

data.Dataset = data.Dataset.map({1:0, 2:1})
data[['Female', 'Male']] = pd.get_dummies(data.Gender)
data.head()
features = data.drop(columns=['Gender', 'Dataset'])

for_viz = data.drop(columns=['Gender', 'Male', 'Female'])
sns.set(style='whitegrid', rc={'figure.figsize':(12,8)})
sns.pairplot(

    for_viz,

    hue='Dataset',

    vars=['Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',

       'Alamine_Aminotransferase', 'Aspartate_Aminotransferase',

       'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']

)
from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(

    features.values,

    data.Dataset,

    stratify=data.Dataset,

    #test_size=0.2,

    random_state=10

)
print(X_train.shape)

print(np.bincount(y_train))

print(np.bincount(y_test))
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline
param_grid = {

    'polynomialfeatures__degree': [1,2,3],

    'logisticregression__penalty': ['l1', 'l2'],

    'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100]

}
pipe = make_pipeline(

    MinMaxScaler(),

    PolynomialFeatures(),

    LogisticRegression()

)
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5).fit(X_train, y_train)
print("Best params: {}".format(grid.best_params_))

print("Best CV score: {}".format(grid.best_score_))

print("Training accuracy: {}".format(grid.score(X_train, y_train)))

print("Test accuracy: {}".format(grid.score(X_test, y_test)))
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve

def accuracy_plots(y_score, plot_name, y_true=y_test):

    fpr, tpr, thresholds = roc_curve(y_test, y_score)

    close_zero = np.argmin(np.abs(thresholds))

    roc_auc = roc_auc_score(y_test, y_score)

    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_score)

    close_zero_pr = np.argmin(np.abs(thresholds_pr))

    

    fig, ax = plt.subplots(2, 1, figsize=(12,14), sharex=True)

    plt.tight_layout()

    plt.subplots_adjust(hspace=0.15, top=0.93)

    fig.suptitle(plot_name, fontsize=14)

    ax[0].plot(fpr, tpr, 'b', label='AUC %0.2f' % roc_auc)

    ax[0].plot(fpr[close_zero], tpr[close_zero], 'o', markersize=12, label="threshold zero", fillstyle="none", c='r', mew=2)

    ax[0].plot([(0,0),(1,1)], c='r', linestyle='--')

    ax[0].set_ylabel('True Positive Rate')

    ax[0].set_xlabel('False Positive Rate')

    ax[0].set_xlim([0,1])

    ax[0].set_ylim([0,1])

    ax[0].legend(loc=4)

    ax[0].set_title('ROC Curve')

    ax[1].plot(precision, recall, 'b')

    ax[1].plot(precision[close_zero_pr], recall[close_zero_pr], 'o', markersize=12, label="threshold zero", fillstyle="none", c='r', mew=2)

    ax[1].set_ylabel('Recall')

    ax[1].set_xlabel('Precision')

    ax[1].set_title('Precision Recall Curve')

    ax[1].legend(loc=4)

    plt.show()
y_score = grid.best_estimator_.decision_function(X_test)
print(confusion_matrix(y_test, grid.best_estimator_.predict(X_test)))

print(classification_report(y_test, grid.best_estimator_.predict(X_test)))
accuracy_plots(y_score, 'Logistic Regression')
plt.figure(figsize=(12,8))

plt.scatter(grid.best_estimator_.decision_function(X_test), y_test, marker='o', c='k')

plt.grid(False)

plt.axvline(0, c='r', label='Decision Boundary')

plt.xlabel('Decision Function')

plt.ylabel('y_test')

plt.title('Possible Improvements to decision boundary')

plt.show()
#class imbalance, class 0 is almost 2.5 times than class 1

data2 = pd.concat([data, data[(data.Dataset == 1)]])
data2.Dataset.value_counts()
data2.head()
X_train, X_test, y_train, y_test = train_test_split(

    data2.drop(columns=['Gender', 'Dataset']),

    data2.Dataset,

    stratify=data2.Dataset,

    random_state=0

)
print(X_train.shape)

print(np.bincount(y_train))

print(np.bincount(y_test))
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5).fit(X_train, y_train)
y_score = grid.best_estimator_.decision_function(X_test)

print("Best params: {}".format(grid.best_params_))

print("Best CV score: {}".format(grid.best_score_))

print("Training accuracy: {}".format(grid.score(X_train, y_train)))

print("Test accuracy: {}".format(grid.score(X_test, y_test)))

print(confusion_matrix(y_test, grid.best_estimator_.predict(X_test)))

print(classification_report(y_test, grid.best_estimator_.predict(X_test)))
accuracy_plots(y_score, 'Logistic Regression')
plt.figure(figsize=(12,8))

plt.scatter(grid.best_estimator_.decision_function(X_test), y_test, marker='o', c='k')

plt.grid(False)

plt.axvline(0, c='r', label='Decision Boundary')

plt.xlabel('Decision Function')

plt.ylabel('y_test')

plt.title('Possible Improvements to decision boundary')

plt.show()