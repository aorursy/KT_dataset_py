import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA, TruncatedSVD

import matplotlib.patches as mpatches

import time



import collections

import scipy.stats as stats



# Other Libraries

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss

from imblearn.metrics import classification_report_imbalanced

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

from collections import Counter

from sklearn.model_selection import KFold, StratifiedKFold

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.head()
df.dtypes
sns.factorplot(x='Class' , kind='count' , data=df , palette=['r','g'])
sns.distplot(df['Amount'],fit=stats.norm)
from sklearn.preprocessing import StandardScaler

df['normAmount'] = StandardScaler().fit_transform(df[['Amount']])

df = df.drop(['Amount'],axis=1)
sns.distplot(df['Time'],fit=stats.norm)
df['normTime'] = StandardScaler().fit_transform(df[['Time']])

df = df.drop(['Time'],axis=1)
df.head()
df.Class.value_counts()
# amount of fraud classes 492 rows.

fraud_df = df.loc[df['Class'] == 1]

non_fraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows

ndf = normal_distributed_df.sample(frac=1, random_state=42)



ndf.head()
sns.factorplot(x='Class' , kind='count' , data=ndf , palette=['r','g'])
f, (g1, g2) = plt.subplots(2, 1, figsize=(24,20))



# Entire (Unbalanced) DataFrame

corr = df.corr()

sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=g1)

g1.set_title("UNbalanced Correlation Matrix \n (not required)", fontsize=14)



# Balanced DataFramw

sub_sample_corr = ndf.corr()

sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=g2)

g2.set_title('SubSample Correlation Matrix \n (required)', fontsize=14)

plt.show()
f, axes = plt.subplots(ncols=4, figsize=(20,4))



sns.boxplot(x="Class", y="V17", data=ndf, palette='Accent', ax=axes[0])

axes[0].set_title('V17 vs Class')



sns.boxplot(x="Class", y="V14", data=ndf, palette='Accent', ax=axes[1])

axes[1].set_title('V14 vs Class')





sns.boxplot(x="Class", y="V12", data=ndf, palette='Accent', ax=axes[2])

axes[2].set_title('V12 vs Class')





sns.boxplot(x="Class", y="V10", data=ndf, palette='Accent', ax=axes[3])

axes[3].set_title('V10 vs Class')



plt.show()
f, axes = plt.subplots(ncols=4, figsize=(20,4))



sns.boxplot(x="Class", y="V11", data=ndf, palette='Accent', ax=axes[0])

axes[0].set_title('V11 vs Class')



sns.boxplot(x="Class", y="V4", data=ndf, palette='Accent', ax=axes[1])

axes[1].set_title('V4 vs Class')





sns.boxplot(x="Class", y="V2", data=ndf, palette='Accent', ax=axes[2])

axes[2].set_title('V2 vs Class')





sns.boxplot(x="Class", y="V19", data=ndf, palette='Accent', ax=axes[3])

axes[3].set_title('V19 vs Class')



plt.show()
# for scaled dataframe

X_ndf = ndf.drop('Class', axis=1)

y_ndf = ndf['Class']

X_train_ndf, X_test_ndf, y_train_ndf, y_test_ndf = train_test_split(X_ndf, y_ndf, test_size=0.2, random_state=42)



# for unbalanced dataframe

X = df.drop('Class', axis=1)

y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Classifier Libraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
X_train_ndf = X_train_ndf.values

X_test_ndf = X_test_ndf.values

y_train_ndf = y_train_ndf.values

y_test_ndf = y_test_ndf.values
# Let's implement simple classifiers



classifiers = {

    "Support Vector Classifier": SVC(),

    "DecisionTreeClassifier": DecisionTreeClassifier()

}
from sklearn.model_selection import cross_val_score

for key, classifier in classifiers.items():

    classifier.fit(X_train_ndf, y_train_ndf)

    training_score = cross_val_score(classifier, X_train_ndf, y_train_ndf, cv=10)

    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
# Use GridSearchCV to find the best parameters.

from sklearn.model_selection import GridSearchCV



# Support Vector Classifier

svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

grid_svc = GridSearchCV(SVC(), svc_params)

grid_svc.fit(X_train_ndf, y_train_ndf)



# SVC best estimator

svc = grid_svc.best_estimator_



# DecisionTree Classifier

tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 

              "min_samples_leaf": list(range(5,7,1))}

grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)

grid_tree.fit(X_train_ndf, y_train_ndf)



# tree best estimator

tree_clf = grid_tree.best_estimator_

# Overfitting Case

svc_score = cross_val_score(svc, X_train_ndf, y_train_ndf, cv=10)

print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')



tree_score = cross_val_score(tree_clf, X_train_ndf, y_train_ndf, cv=10)

print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')
y_pred_ndf=grid_tree.predict(X_test_ndf)

from sklearn import metrics

metrics.accuracy_score(y_pred_ndf, y_test_ndf)
tree_clf
grid_tree.fit(X_train,y_train)
y_pred=grid_tree.predict(X_test)
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report

y_pred = grid_tree.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)



roc_auc = auc(fpr,tpr)



# Plot ROC

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()

roc_auc
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score

cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred, normalize = True)

precision = precision_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)
print("\nconfusion_matrix{}\n".format(cm))

print("accuracy_score{}\n".format(accuracy))

print("precision_score{}\n".format(precision))

print("f1_score{}\n".format(f1))