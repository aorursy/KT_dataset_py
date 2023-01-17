# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.manifold import TSNE

from sklearn.decomposition import PCA, TruncatedSVD

import matplotlib.patches as mpatches

import time
data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

print (data.info())

# print (data.head(10))
print (data[data['Class'] == 1])
g = data.corr()

sns.heatmap(g)
count_0, count_1 = data.Class.value_counts()



data_class_0 = data[data.Class == 0] 

data_class_1 = data[data.Class == 1]



data_sampled = data_class_0.sample(count_1)

data_sampled = pd.concat([data_sampled, data_class_1], axis=0)

data_sampled.Class.value_counts().plot(kind='bar', title='Count (Class)');

fig, ax = plt.subplots(figsize=(10,8))     

g = data_sampled.corr()

sns.heatmap(g, cmap = 'coolwarm_r', annot_kws={'size':60},ax = ax)
from scipy.stats import norm



new_df = data_sampled



f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 6))



v14_fraud_dist = new_df['V14'].loc[new_df['Class'] == 1].values

sns.distplot(v14_fraud_dist,ax=ax1, fit=norm, color='#FB8861')

ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)



v12_fraud_dist = new_df['V12'].loc[new_df['Class'] == 1].values

sns.distplot(v12_fraud_dist,ax=ax2, fit=norm, color='#56F9BB')

ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)





v10_fraud_dist = new_df['V10'].loc[new_df['Class'] == 1].values

sns.distplot(v10_fraud_dist,ax=ax3, fit=norm, color='#C5B3F9')

ax3.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)



plt.show()

from collections import Counter



def detect_outliers(df, features):

    outlier_indices = []

    

    for col in features:

        Q1 = np.percentile(df[col], 25)

        

        Q3 = np.percentile(df[col], 75)

        

        IQR = Q3 - Q1

        outlier_step = 2.5 * IQR

    

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > 1 )

    return multiple_outliers

    

    

outliers_to_drop = detect_outliers(data_sampled, ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13',

                                              'V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26',

                                              'V27','V28'])   

new_df = data_sampled.drop(outliers_to_drop, axis = 0).reset_index(drop=True)
from scipy.stats import norm



f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 6))



v14_fraud_dist = new_df['V14'].loc[new_df['Class'] == 1].values

sns.distplot(v14_fraud_dist,ax=ax1, fit=norm, color='#FB8861')

ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)



v12_fraud_dist = new_df['V12'].loc[new_df['Class'] == 1].values

sns.distplot(v12_fraud_dist,ax=ax2, fit=norm, color='#56F9BB')

ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)





v10_fraud_dist = new_df['V10'].loc[new_df['Class'] == 1].values

sns.distplot(v10_fraud_dist,ax=ax3, fit=norm, color='#C5B3F9')

ax3.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)



plt.show()



X = new_df.drop('Class', axis=1)

y = new_df['Class']





# T-SNE Implementation

t0 = time.time()

X_reduced_tsne = TSNE(n_components=3, random_state=42).fit_transform(X.values)

t1 = time.time()

print("T-SNE took {:.2} s".format(t1 - t0))



# PCA Implementation

t0 = time.time()

X_reduced_pca = PCA(n_components=3, random_state=42).fit_transform(X.values)

t1 = time.time()

print("PCA took {:.2} s".format(t1 - t0))



# TruncatedSVD

t0 = time.time()

X_reduced_svd = TruncatedSVD(n_components=3, algorithm='randomized', random_state=42).fit_transform(X.values)

t1 = time.time()

print("Truncated SVD took {:.2} s".format(t1 - t0))
from sklearn.model_selection import train_test_split

X = new_df.drop('Class', axis=1)

y = new_df['Class']



X_train, X_test, y_train, y_test = train_test_split(X,y,random_state= 2)
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import collections



classifiers = {

    "LogisticRegression": LogisticRegression(),

    "Support Vector Classifier": SVC(),

    "KNN": KNeighborsClassifier(),

    "DecisionTree": DecisionTreeClassifier(),

}
from sklearn.model_selection import cross_val_score





for key, classifier in classifiers.items():

    classifier.fit(X_train, y_train)

    training_score = cross_val_score(classifier, X_train, y_train, cv=10)

    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
# Use GridSearchCV to find the best parameters.

from sklearn.model_selection import GridSearchCV



#LogisiticRegression 

logreg_par = {'C':[0.1, 0.3, 1, 3, 10, 30]}



logreg_grid = GridSearchCV(LogisticRegression(), logreg_par)

logreg_grid.fit(X_train, y_train)

logreg_bestfit = logreg_grid.best_estimator_

print (logreg_bestfit)



svc_par = {'gamma':[0.1, 0.3, 1, 3, 10, 30],'C':[0.1, 0.3, 1, 3, 10, 30]}



svc_grid = GridSearchCV(SVC(), svc_par)

svc_grid.fit(X_train, y_train)

svc_bestfit = svc_grid.best_estimator_

print (svc_bestfit)



knn_par = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}



knn_grid = GridSearchCV(KNeighborsClassifier(), knn_par)

knn_grid.fit(X_train, y_train)

knn_bestfit = knn_grid.best_estimator_

print (knn_bestfit)



tree_par = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 

              "min_samples_leaf": list(range(5,7,1))}

tree_grid =GridSearchCV(DecisionTreeClassifier(), tree_par)

tree_grid.fit(X_train,y_train)

tree_bestfit = tree_grid.best_estimator_

print (tree_bestfit)

classifiers = {

    "LogisticRegression": logreg_bestfit,

    "Support Vector Classifier": svc_bestfit,

    "KNN": knn_bestfit,

    "DecisionTree": tree_bestfit,

}

for key, classifier in classifiers.items():

    classifier.fit(X_train, y_train)

    training_score = cross_val_score(classifier, X_train, y_train, cv=10)

    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
from sklearn.model_selection import ShuffleSplit 

from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, ax,label, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5),):

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax.set_title( label+" Curve", fontsize=14)

    ax.set_xlabel('Training size (m)')

    ax.set_ylabel('Score')

    ax.grid(True)

    ax.legend(loc="best")

    
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

def plot_curve(e1,e2,e3,e4):

    f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(20,14), sharey=True)

    lst = [[e1,ax1],[e2,ax2],[e3,ax3],[e4,ax4]]

    plot_learning_curve(e1, X_train, y_train, ax1,"logistic regression" ,(0.87, 1.01), cv=cv, n_jobs=10)

    plot_learning_curve(e2, X_train, y_train, ax2,"svc" ,(0.87, 1.01), cv=cv, n_jobs=10)

    plot_learning_curve(e3, X_train, y_train, ax3,"knn",(0.87, 1.01), cv=cv, n_jobs=10)

    plot_learning_curve(e4, X_train, y_train, ax4,"decision tree",(0.87, 1.01), cv=cv, n_jobs=10)

plot_curve(logreg_bestfit, svc_bestfit, knn_bestfit, tree_bestfit)
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score,recall_score, f1_score





X = data.drop('Class', axis=1)

y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state= 2)



fig = plt.figure(figsize=(12,6))



logreg_pred = logreg_bestfit.predict(X_train)



precision, recall, threshold = precision_recall_curve(y_train, logreg_pred)



pre = precision_score(y_train, logreg_pred)

rec = recall_score(y_train, logreg_pred)

score = logreg_bestfit.score(X_test, y_test)

f1 = f1_score(y_train, logreg_pred)



plt.step(recall, precision, color='#004a93', alpha=0.2,

         where='post')

plt.fill_between(recall, precision, step='post', alpha=0.2,

                 color='#48a6ff')



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

# plt.title('UnderSampling Precision-Recall curve: \n Average Precision-Recall Score ={0:0.2f}'.format(

#           undersample_average_precision), fontsize=16)





print (confusion_matrix(y_train, logreg_pred))

print ("precsion:",pre,"recall",rec)

print ("f1 score:",f1)

print ("accuracy score",score)