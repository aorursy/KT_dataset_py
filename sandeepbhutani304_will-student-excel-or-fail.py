import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
sns.set_color_codes("pastel")
%matplotlib inline
import os
os.listdir('../input')
# I'm use only student-mat.csv
data = pd.read_csv('../input/student-mat.csv')
print("G3 range: Min={}, Max={}".format(data["G3"].min(), data["G3"].max()))
data.head(5)

def create_g3_class(data):
    return ["Fail", "Medium", "Good"][0 if data["G3"] <= 5 else 1 if data["G3"] <= 15 else 2]

data["G3_class"] = data.apply(lambda row: create_g3_class(row), axis=1)
data.head()
data.info()
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb

from sklearn.model_selection import cross_val_score
y = data['G3_class']
X = data.drop(['G3', 'G3_class'], axis=1)
X = pd.get_dummies(X)
names = ['RandomForestClassifier', 'NaiveBayes' , 'DecisionTreeClassifier', 'XGBClassifier']

clf_list = [RandomForestClassifier(),
            MultinomialNB(),
            DecisionTreeClassifier(),
           xgb.XGBClassifier()]
clf_scores = {}
for name, clf in zip(names, clf_list):
    clf_scores[name]= cross_val_score(clf, X, y, cv=5).mean()
    print(name, end=': ')
    print(clf_scores[name])
best_classifier = sorted(clf_scores, key=clf_scores.get, reverse=True)[0]
best_classifier
clf = clf_list[names.index(best_classifier)]
clf.fit(X, y)
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X.shape[1]):
    if(importances[indices[f]] >= 0.01):
        print("%d. Feature %s (%f)" % (f + 1, X.columns.values[indices[f]], importances[indices[f]]))
X = data.drop(['G3', 'G2', 'G1', 'G3_class'], axis=1)
X = pd.get_dummies(X)
clf_scores = {}
for name, clf in zip(names, clf_list):
    clf_scores[name]= cross_val_score(clf, X, y, cv=5).mean()
    print(name, end=': ')
    print(clf_scores[name])
best_classifier = sorted(clf_scores, key=clf_scores.get, reverse=True)[0]
best_classifier
clf = clf_list[names.index(best_classifier)]
clf.fit(X, y)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X.shape[1]):
    if(importances[indices[f]] >= 0.01):
        print("%d. Feature %s (%f)" % (f + 1, X.columns.values[indices[f]], importances[indices[f]]))
from sklearn.model_selection import train_test_split
X = data.drop(['G3_class'], axis=1)
X = pd.get_dummies(X)     #Convert to categorical
y = data['G3_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, stratify=y)
X_train= X_train.drop(['G3'], axis=1)
import copy
X_test_withG3 = copy.deepcopy(X_test)    #will be used in end to display actual G3 score
X_test= X_test.drop(['G3'], axis=1)
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
clf = clf_list[names.index(best_classifier)]
print("using classifer: %s"%best_classifier)
# clf = xgb.XGBClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)

np.set_printoptions(precision=2)
# print(clf.classes_)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=clf.classes_, title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=clf.classes_, normalize=True, title='Normalized confusion matrix')

plt.show()
plt.figure()
plt.boxplot(data['G3'], notch=True, sym='gD', vert=False)
plt.title('G3 (final grade) score distribution in dataset')
plt.show()
p = sns.countplot(data['G3_class']).set_title('G3 class distribution')
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import CondensedNearestNeighbour, AllKNN, OneSidedSelection, RandomUnderSampler
from imblearn.ensemble import BalanceCascade, EasyEnsemble

from collections import Counter
# X_resampled, y_resampled = ADASYN().fit_resample(X_train, y_train)
X_resampled, y_resampled = AllKNN(sampling_strategy=['Medium']).fit_resample(X_train, y_train)
# X_resampled, y_resampled = SMOTETomek().fit_resample(X_train, y_train)  #sampling_strategy='minority'
# X_resampled, y_resampled = EasyEnsemble().fit_resample(X_train, y_train)
# X_resampled = X_resampled[0] ; y_resampled = y_resampled[0]
print(sorted(Counter(y_resampled).items()))
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X_resampled) #pd.get_dummies(data)
principalDf_train = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
principalComponentstest = pca.fit_transform(X_test) #pd.get_dummies(data)
principalDf_test = pd.DataFrame(data = principalComponentstest
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])

clf = clf_list[names.index(best_classifier)]
print("Using classifier: %s"%best_classifier)
# clf = xgb.XGBClassifier()
# clf = DecisionTreeClassifier()  #Using for demo and consistency
clf.fit(principalDf_train, y_resampled)

y_pred = clf.predict(principalDf_test)
y_pred_prob = clf.predict_proba(principalDf_test)
cnf_matrix = confusion_matrix(y_test, y_pred)

np.set_printoptions(precision=2)
# print(clf.classes_)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=clf.classes_, title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=clf.classes_, normalize=True, title='Normalized confusion matrix')

plt.show()
y_test_list = list(y_test)
X_test_withG3_list = list(X_test_withG3['G3'])
for idx, item in enumerate(y_pred):
    if(item == 'Fail'):
        print("Student {} \t [Actual Failed?: {}  \tG3: {}]".format(idx, y_test_list[idx], X_test_withG3_list[idx]))