import pandas as pd

import numpy as np



import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid", color_codes=True)

%matplotlib inline
df = pd.read_csv('../input/data.csv', header=0)

df.describe()
df.isnull().sum()
df = df.drop('Unnamed: 32', axis=1)
map_diagnosis = {'M':1, 'B':0}

df['diagnosis'] = df['diagnosis'].map(map_diagnosis)



sns.countplot(x='diagnosis', data=df)
correlations = df.corr()

k = 12

cols = correlations.nlargest(k, 'diagnosis')['diagnosis'].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
sns.jointplot('concave points_worst', 'perimeter_worst', data=df)

plt.show()
#We will try to reduce the dimension thanks to a PCA
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.cross_validation import train_test_split



y = df['diagnosis'].values

df = df.drop(['diagnosis'], axis=1)

X = df.iloc[:, 1:]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



std = StandardScaler()

X_train = std.fit_transform(X_train)

X_test = std.transform(X_test)



pca = PCA()

pca.fit(X_train)

explained_variance = pca.explained_variance_ratio_

cumsum_explained_variance = explained_variance.cumsum()



plt.plot(range(len(cumsum_explained_variance)), cumsum_explained_variance)

plt.show()
pca = PCA(n_components=10)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)
from sklearn.linear_model import LogisticRegression

from sklearn.grid_search import GridSearchCV



param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]



lr = LogisticRegression(random_state=0, penalty='l2')



param_grid = [{'C':param_range}]



gs = GridSearchCV(estimator=lr,

                 param_grid=param_grid,

                 scoring='roc_auc', 

                 cv=10)



gs.fit(X_train, y_train)
best_estimator = gs.best_estimator_
from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score





y_lr_train_prediction = best_estimator.predict(X_train)





print ('ROC AUC SCORE for Logistic regression TRAIN %.3f ' %roc_auc_score(y_true=y_train, y_score=y_lr_train_prediction))

print ('PRECISION SCORE for Logistic regression TRAIN %.3f ' %precision_score(y_true=y_train, y_pred=y_lr_train_prediction))

print ('RECALL SCORE for Logistic regression TRAIN %.3f ' %recall_score(y_true=y_train, y_pred=y_lr_train_prediction))

print ('F1-SCORE SCORE for Logistic regression %.3f TRAIN  \n' %f1_score(y_true=y_train, y_pred=y_lr_train_prediction))







y_lr_test_prediction = best_estimator.predict(X_test)



print ('ROC AUC SCORE for Logistic regression %.3f ' %roc_auc_score(y_true=y_test, y_score=y_lr_test_prediction))

print ('PRECISION SCORE for Logistic regression %.3f ' %precision_score(y_true=y_test, y_pred=y_lr_test_prediction))

print ('RECALL SCORE for Logistic regression %.3f ' %recall_score(y_true=y_test, y_pred=y_lr_test_prediction))

print ('F1-SCORE SCORE for Logistic regression %.3f ' %f1_score(y_true=y_test, y_pred=y_lr_test_prediction))





from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train, y_train)



y_knn_train_prediction = knn.predict(X_train)



print ('ROC AUC SCORE for KNN TRAIN %.3f ' %roc_auc_score(y_true=y_train, y_score=y_knn_train_prediction))

print ('PRECISION SCORE for KNN TRAIN %.3f ' %precision_score(y_true=y_train, y_pred=y_knn_train_prediction))

print ('RECALL SCORE for KNN TRAIN %.3f ' %recall_score(y_true=y_train, y_pred=y_knn_train_prediction))

print ('F1-SCORE SCORE for KNN TRAIN %.3f \n' %f1_score(y_true=y_train, y_pred=y_knn_train_prediction))



y_knn_test_prediction = knn.predict(X_test)



print ('ROC AUC SCORE for KNN  %.3f ' %roc_auc_score(y_true=y_test, y_score=y_knn_test_prediction))

print ('PRECISION SCORE for KNN  %.3f ' %precision_score(y_true=y_test, y_pred=y_knn_test_prediction))

print ('RECALL SCORE for KNN  %.3f ' %recall_score(y_true=y_test, y_pred=y_knn_test_prediction))

print ('F1-SCORE SCORE for KNN  %.3f ' %f1_score(y_true=y_test, y_pred=y_knn_test_prediction))

from sklearn.svm import SVC



param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000]

param_grid = [

    {

        'C': param_range,

        'kernel' : ['linear']

    },

    {

        'C': param_range,

        'kernel' : ['rbf'],

        'gamma': param_range

    }

]



svm = SVC(random_state=0)



gs_svc = GridSearchCV(estimator=svm,

                     param_grid=param_grid,

                     scoring='roc_auc',

                     cv=10,

                     n_jobs=1)



gs_svc.fit(X_train, y_train)
gs_svc.best_params_
y_radial_svc_train_prediction = gs_svc.best_estimator_.predict(X_train)



print ('ROC AUC SCORE for SVC TRAIN %.3f ' %roc_auc_score(y_true=y_train, y_score=y_radial_svc_train_prediction))

print ('PRECISION SCORE for SVC TRAIN %.3f ' %precision_score(y_true=y_train, y_pred=y_radial_svc_train_prediction))

print ('RECALL SCORE for SVC TRAIN %.3f ' %recall_score(y_true=y_train, y_pred=y_radial_svc_train_prediction))

print ('F1-SCORE SCORE for SVC TRAIN %.3f \n' %f1_score(y_true=y_train, y_pred=y_radial_svc_train_prediction))



y_radial_svc_test_prediction = gs_svc.best_estimator_.predict(X_test)



print ('ROC AUC SCORE for SVC  %.3f ' %roc_auc_score(y_true=y_test, y_score=y_radial_svc_test_prediction))

print ('PRECISION SCORE for SVC  %.3f ' %precision_score(y_true=y_test, y_pred=y_radial_svc_test_prediction))

print ('RECALL SCORE for SVC  %.3f ' %recall_score(y_true=y_test, y_pred=y_radial_svc_test_prediction))

print ('F1-SCORE SCORE for SVC  %.3f ' %f1_score(y_true=y_test, y_pred=y_radial_svc_test_prediction))
from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(max_depth=4, random_state=0)

tree.fit(X_train, y_train)



y_tree_train_prediction = tree.predict(X_train)



print ('ROC AUC SCORE for SVC TRAIN %.3f ' %roc_auc_score(y_true=y_train, y_score=y_tree_train_prediction))

print ('PRECISION SCORE for SVC TRAIN %.3f ' %precision_score(y_true=y_train, y_pred=y_tree_train_prediction))

print ('RECALL SCORE for SVC TRAIN %.3f ' %recall_score(y_true=y_train, y_pred=y_tree_train_prediction))

print ('F1-SCORE SCORE for SVC TRAIN %.3f \n' %f1_score(y_true=y_train, y_pred=y_tree_train_prediction))



y_tree_test_prediction = tree.predict(X_test)



print ('ROC AUC SCORE for SVC  %.3f ' %roc_auc_score(y_true=y_test, y_score=y_tree_test_prediction))

print ('PRECISION SCORE for SVC  %.3f ' %precision_score(y_true=y_test, y_pred=y_tree_test_prediction))

print ('RECALL SCORE for SVC  %.3f ' %recall_score(y_true=y_test, y_pred=y_tree_test_prediction))

print ('F1-SCORE SCORE for SVC  %.3f ' %f1_score(y_true=y_test, y_pred=y_tree_test_prediction))