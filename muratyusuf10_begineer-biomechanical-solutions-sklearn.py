# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import accuracy_score 

from sklearn.model_selection import cross_val_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





import warnings

warnings.filterwarnings("ignore")





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

df.columns

df.info()
df.head()
data_train,data_labels = df.loc[:,df.columns != 'class'], df.loc[:,'class']

color_list = ['red' if i =='Abnormal' else 'blue' for i in df.loc[:,'class']]

pd.plotting.scatter_matrix(df.loc[:,df.columns != 'class'],

                          c=color_list,

                          figsize=[15,15],

                          diagonal='hist',

                          alpha=0.5,

                          s=200,

                          marker='*',

                          edgecolor='black')

plt.show()



#DATAITEAM
data_labels.value_counts()
#data_labelsda ki verileri saydıran ve görselleştiren plot. 

sns.countplot(data_labels, data=df)

data_labels = [1 if i == "Abnormal" else 0 for i in data_labels]
#check the labels

data_labels
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_train,data_labels,test_size=0.3,random_state=1)

#KNN

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier()

knn_model.fit(x_train,y_train)

print("KNN score:",knn_model.score(x_test,y_test)) #predicted and accuracy score
#Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print("Naive Bayes score:",nb.score(x_test,y_test)) #predicted and accuracy score
#Random Forest Classifier 

from sklearn.ensemble import RandomForestClassifier

rfc_model = RandomForestClassifier(n_estimators=100,random_state=42)

rfc_model.fit(x_train,y_train)

print("RFC score:",rfc_model.score(x_test,y_test))
#Logistic Regression 

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()

lr_model.fit(x_train,y_train)

lr_predicted = lr_model.predict(x_test)

print("LR score:", accuracy_score(y_test,lr_predicted))
#Super Vector Machinee

from sklearn.svm import SVC

svm_model = SVC(random_state=42)

svm_model.fit(x_train,y_train)

print("SVM score:",svm_model.score(x_test,y_test))

#XGBOOST

from xgboost import XGBClassifier



xgb = XGBClassifier()

xgb.fit(x_train,y_train)

xgb_predicted = xgb.predict(x_test)

print('XGBoost',accuracy_score(y_test, xgb_predicted))

from sklearn.preprocessing import StandardScaler, Normalizer



norm = Normalizer()

norm_data_train = norm.fit_transform(data_train)
#Data set values changed 0-1 

norm_data_train
#KNN 

from sklearn.neighbors import KNeighborsClassifier

knn_model_norm =KNeighborsClassifier(n_neighbors=3)



print("normalize knn:",cross_val_score(knn_model_norm, norm_data_train,data_labels,cv=10).mean())
# NAIVE BAYES

from sklearn.naive_bayes import GaussianNB

nb_model_norm = GaussianNB()



print("normalize naive bayes:", cross_val_score(nb_model_norm,norm_data_train,data_labels, cv=10).mean())
#Random Forest Classifier 

from sklearn.ensemble import RandomForestClassifier

rfc_model_norm = RandomForestClassifier(n_estimators=100,random_state=42)



print("RFC score:",cross_val_score(rfc_model_norm,norm_data_train,data_labels, cv=10).mean())
#Logistic Regression 

from sklearn.linear_model import LogisticRegression

lr_model_norm = LogisticRegression()



print("LR score:", cross_val_score(lr_model_norm,norm_data_train,data_labels, cv=10).mean())
#Super Vector Machinee

from sklearn.svm import SVC

svm_model_norm = SVC(random_state=42)



print("SVM score:",cross_val_score(svm_model_norm,norm_data_train,data_labels,cv=10).mean())
#XGBOOST

from xgboost import XGBClassifier

xgb_norm = XGBClassifier()



print('XGBoost',cross_val_score(xgb_norm,norm_data_train,data_labels,cv=10).mean())
#PCA

from sklearn.decomposition import PCA

pca_model = PCA(n_components=6)



pca_data = pca_model.fit_transform(data_train)

explained_variance = pca_model.explained_variance_ratio_

print(explained_variance)
#KNN 

from sklearn.neighbors import KNeighborsClassifier

knn_model_pca =KNeighborsClassifier(n_neighbors=3)

print("KNN with PCA score : ",cross_val_score(knn_model_pca, pca_data, data_labels, cv=10).mean())
#%% Naive Bayes 

from sklearn.naive_bayes import GaussianNB

nb_model_pca = GaussianNB()



print('Naive Bayes with PCA score:',cross_val_score(nb_model_pca,pca_data, data_labels, cv=10).mean())
# %% Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rfc_model_pca = RandomForestClassifier()



print("Random Forest with PCA score:", cross_val_score(rfc_model_pca,pca_data,data_labels, cv = 10).mean())
#%% Logistic Regression

from sklearn.linear_model import LogisticRegression

lr_model_pca = LogisticRegression()



print("Logistic Regression with PCA score:",cross_val_score(lr_model_pca,pca_data,data_labels, cv=10).mean())
#%% Super Vector Machine

from sklearn.svm import SVC

svm_model_pca = SVC()



print("Super Vector Machine:",cross_val_score(svm_model_pca,pca_data,data_labels,cv=10).mean())
#XGBOOST

from xgboost import XGBClassifier

xgb_model_pca = XGBClassifier()



print('XGBoost',cross_val_score(xgb_model_pca,pca_data,data_labels,cv=10).mean())
#Grid Search CV and Gaussian Mixture 

#%% 

from sklearn.mixture import GaussianMixture

from sklearn.model_selection import GridSearchCV, cross_val_score



lowest_bic = np.infty

bic = []

n_components_range = range(1,7)

cv_types = ['spherical','tied','diag','full']

for cv_type in cv_types:

    for n_components in n_components_range:

        gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type)

        gmm.fit(data_train)

        bic.append(gmm.aic(data_train))

        if bic[-1] < lowest_bic:

            lowest_bic = bic[-1]

            best_gmm = gmm

best_gmm.fit(data_train)

gmm_train = best_gmm.predict_proba(data_train)



rfc_model3 = RandomForestClassifier(random_state=42)

knn_model3 = KNeighborsClassifier()

svm_model3 = SVC()

n_estimators = [10, 50, 100, 200,500]

max_depth = [1, 5, 10, 20]

param_grid = dict(n_estimators=n_estimators,max_depth=max_depth)



grid_search_rfc = GridSearchCV(estimator=rfc_model3,param_grid=param_grid, cv =10, scoring='accuracy', n_jobs=1).fit(gmm_train,data_labels)

rfc_best = grid_search_rfc.best_estimator_

print('Random Forest Best Score:',grid_search_rfc.best_score_)

print('Random Forest Best Param:',grid_search_rfc.best_params_)

print('Random Forest Best Accuracy:',cross_val_score(rfc_best,gmm_train, data_labels,cv=10).mean())
score_list = []

for each in range(1,25):

    knn2 =KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,25),score_list)

plt.xlabel("K values")

plt.ylabel("Accuary")

plt.show