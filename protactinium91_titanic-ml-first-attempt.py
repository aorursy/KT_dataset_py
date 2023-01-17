





#Importing necessary libraries

%matplotlib inline

import matplotlib as mpl

from matplotlib import colors

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='ticks')

import pandas as pd

import numpy as np

import glob

import os

import time

import sys

import category_encoders as ce

#Various sklearn utilities and metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from sklearn.externals import joblib

from sklearn.ensemble import GradientBoostingRegressor

from yellowbrick.regressor import PredictionError, ResidualsPlot





#Various sklearn utilities and metrics

from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, ShuffleSplit

from sklearn.metrics import classification_report, auc, roc_curve, confusion_matrix

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.externals import joblib



from sklearn.ensemble import (RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier)



#Import categorical encodin

#g library

import category_encoders as ce



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
basedata = pd.read_csv("../input/train.csv")

basedata.Embarked.unique()

emb = basedata.Embarked



emb[pd.isnull(basedata.Embarked)]  = 'NaN'

emb.unique()

basedata.Embarked = emb

basedata.head()
# check data



basedata.PassengerId.isnull().values.any()
basedata.Survived.isnull().values.any()
round(basedata.Age.mean(),2)
basedata.Age.isnull().values.any()

## there are NaN age values

## replace with mean



ag = basedata.Age



ag[pd.isnull(basedata.Age)]  = round(basedata.Age.mean(),2)

ag.unique()
basedata.Age = ag


basedata.SibSp.isnull().values.any()

basedata.Parch.isnull().values.any()
basedata.Fare.isnull().values.any()
sx_Label = LabelEncoder()

sex_labels = sx_Label.fit_transform(basedata['Sex'])

sex_mapping = {index: label for index, label in enumerate(sx_Label.classes_)}

print("Sex label mapping:", sex_mapping)

basedata['Sex'] = sex_labels
emb_Label = LabelEncoder()

embarked_labels = emb_Label.fit_transform(basedata['Embarked'])

embarked_mapping = {index: label for index, label in enumerate(emb_Label.classes_)}

print("Embarked label mapping:", embarked_mapping)

basedata['Embarked'] = embarked_labels
basedata.head()
ml_data = basedata.iloc[:,[0,2,4,5,6,7,9,1]]

ml_data.head()
#Feature variables



X= ml_data.iloc[:, [1,2,3]]





#Target variable



y = ml_data.iloc[:, 7]
X
binary = ce.BinaryEncoder(verbose=1, drop_invariant=True, 

                          cols=['Sex']).fit(X, y)

Results_bin = binary.transform(X)




y_classes = np.empty((len(y),1))
y_classes
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=8, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure(figsize=(15,10))

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, scoring='accuracy', train_sizes=train_sizes, shuffle=True)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
rndForest = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=2, max_depth=None, 

                                  min_samples_leaf=1, max_leaf_nodes=None)



X_train, X_test, y_train, y_test = train_test_split(Results_bin, y, test_size=0.5, random_state=1337)

y_train = y_train.ravel().astype('int')

y_test = y_test.ravel().astype('int')



rndForest.fit(X_train,y_train)

y_pred_bin = rndForest.predict(X_test)



#Show classification report

print(classification_report(y_test, y_pred_bin, labels=(0,1)))



#Print ROC curve

fpr = dict()

tpr = dict()

roc_auc = dict()

fpr, tpr, thresholds = roc_curve(y_test, y_pred_bin)

roc_auc = auc(fpr, tpr)



plt.figure(figsize=(10,5))

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic curve for Random Forest Classifier and Binary categorical feature encoding')

plt.legend(loc="lower right")

plt.savefig('ROC_Binary.png', dpi=200)

plt.show()



cv = ShuffleSplit(n_splits=10, test_size=0.8, random_state=1337)



#Print learning curve

plot_learning_curve(rndForest, 'Learning curve for Random Forest Classifier and Binary categorical feature encoding', 

                    Results_bin.values, y, ylim=None, cv=cv, train_sizes=np.linspace(.1, 1.0, 10))

plt.savefig('LearningCurve_Binary.png', dpi=200)




cm = confusion_matrix(y_test, y_pred_bin)





sns.heatmap(cm, fmt='d', annot=True,annot_kws={"size": 16})# font size

plt.show()