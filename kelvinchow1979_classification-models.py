# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
##Import

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression, lars_path

from sklearn import svm

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score

from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, f1_score, make_scorer, confusion_matrix

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.dummy import DummyClassifier

from sklearn.inspection import permutation_importance

from xgboost.sklearn import XGBRegressor

from sklearn.naive_bayes import GaussianNB
#Data Import

data=pd.read_csv('../input/churn-modelling/Churn_Modelling.csv',index_col='RowNumber')
data.head()
data.Geography.value_counts()
data.shape
data.info()
mask=np.zeros_like(data.corr())

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(13, 5))

sns.heatmap(data.corr(), mask=mask, cmap='seismic', annot=True, vmin=-1, vmax=1,)
sns.pairplot(data, height=1.2, aspect=1.5)
data.columns
#The Target column

y=data['Exited']

y.head()
#Remove unnecessary columns for X

X=data.drop(['CustomerId','Surname','Exited'],axis='columns')

X.head()
#Transform categorical date to numerical with one-hot

X_numerical=pd.get_dummies(X)

X_numerical.head()
#Drop each of categorical first columns

X_num=X_numerical.drop(['Geography_Spain','Gender_Male'],axis='columns')

X_num.head()
#Scale all the numerical columns

scaler=StandardScaler()

X_scaled=scaler.fit_transform(X_num)

X_scaled
sns.pairplot(pd.DataFrame(X_scaled), height=1.2, aspect=1.5)
#Train Test Split

X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y, test_size=.2, random_state=50)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=.25, random_state=50)
#Lasso Path

print("Computing regularization path using the LARS ...")

alphas, _, coefs = lars_path(X_scaled, y.values, method='lasso')
#Plot Lasso Path

xx = np.sum(np.abs(coefs.T), axis=1)

xx /= xx[-1]



plt.figure(figsize=(10,10))

plt.plot(xx, coefs.T)

ymin, ymax = plt.ylim()

plt.vlines(xx, ymin, ymax, linestyle='dashed')

plt.xlabel('|coef| / max|coef|')

plt.ylabel('Coefficients')

plt.title('LASSO Path')

plt.axis('tight')

plt.legend(X_num.columns)

plt.show()
#Gaussian Naive Bayes model

nb = GaussianNB()

nb.fit(X_train_val, y_train_val)

predictionnb=nb.predict(X_test)

f1_score(y_test, predictionnb), precision_score(y_test, predictionnb), recall_score(y_test, predictionnb)
#Logistic Regression Model

logit = LogisticRegression(solver= 'liblinear', C=1)

logit.fit(X_train_val, y_train_val)

predictionL = logit.predict(X_test)

confusion_matrix(y_test, predictionL), f1_score(y_test, predictionL), precision_score(y_test, predictionL), recall_score(y_test, predictionL)
#Logistic GridSearchCV

solver_list = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']

parameters = dict(solver=solver_list)

logit = LogisticRegression(random_state=34, C=1)

f1score=make_scorer(f1_score)

Grid1 = GridSearchCV(logit, parameters, scoring=f1score, cv=5)

Grid1.fit(X_train_val, y_train_val)

Grid1.best_params_, Grid1.best_score_
Grid1.cv_results_
scores = Grid1.cv_results_['mean_test_score']



for score, solver, in zip(scores, solver_list):

    print(f"{solver}: {score:.3f}")
#K Nearest Neighbor model

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train_val, y_train_val)

predictionKn=knn.predict(X_test)

f1_score(y_test, predictionKn),precision_score(y_test, predictionKn), recall_score(y_test, predictionKn)
#KNN GridSearchCV

knn_params={

    'n_neighbors':[3,5,7,9,11],

    'weights':['uniform','distance'],

    'algorithm':['ball_tree','kd_tree','brute']

}

knng=GridSearchCV(KNeighborsClassifier(),knn_params,verbose=1,cv=5)

GridK=knng.fit(X_train_val, y_train_val)

predictionKnn=knng.predict(X_test)

f1_score(y_test, predictionKnn), precision_score(y_test, predictionKnn), recall_score(y_test, predictionKnn)
GridK.best_params_, GridK.best_score_
#Linear SVC Model

sv=svm.LinearSVC()

sv.fit(X_train_val, y_train_val)

predictionSV=sv.predict(X_test)

f1_score(y_test, predictionSV),precision_score(y_test, predictionSV), recall_score(y_test, predictionSV)
#SVC Radial Model

svRa=svm.SVC(kernel='rbf', gamma="scale")

svRa.fit(X_train_val, y_train_val)

predictionSVra=svRa.predict(X_test)

f1_score(y_test, predictionSVra), precision_score(y_test, predictionSVra), recall_score(y_test, predictionSVra)
#SVC Poly Model

svPoly=svm.SVC(kernel='poly', degree=4, gamma="scale")

svPoly.fit(X_train_val, y_train_val)

predictionSVpoly=svPoly.predict(X_test)

f1_score(y_test, predictionSVpoly), precision_score(y_test, predictionSVpoly), recall_score(y_test, predictionSVpoly)
#Random Forest Model

rand=RandomForestClassifier(n_estimators=300)

rand.fit(X_train_val, y_train_val)

predictionRand=rand.predict(X_test)

confusion_matrix(y_test, predictionRand), f1_score(y_test, predictionRand), precision_score(y_test, predictionRand), recall_score(y_test, predictionRand)
#Random Forest with adjusted threshold

thress=0.39

predictionRandt = (rand.predict_proba(X_test)[:,1] > thress)

print("Threshold of {:6.2f}:".format(thress))

print("Precision: {:6.4f},   Recall: {:6.4f},   F1: {:6.4f}".format(precision_score(y_test, predictionRandt), 

                                                     recall_score(y_test, predictionRandt),f1_score(y_test, predictionRandt)))
confusion_matrix(y_test, predictionRandt)
#XGBoost Model

xgbr = XGBRegressor(n_estimators=300, learning_rate=0.01)

xgbr.fit(X_train_val, y_train_val)

predictionXgbr=xgbr.predict(X_test)

confusion_matrix(y_test, predictionXgbr.round()), f1_score(y_test, predictionXgbr.round()), precision_score(y_test, predictionXgbr.round()), recall_score(y_test, predictionXgbr.round())
#XGBoost GridSearchCV

xgb1 = XGBRegressor()

parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower

              'objective':['reg:linear'],

              'learning_rate': [.03, 0.05, .07], #so called `eta` value

              'max_depth': [5, 6, 7],

              'min_child_weight': [4],

              'silent': [1],

              'subsample': [0.7],

              'colsample_bytree': [0.7],

              'n_estimators': [500]}



xgb_grid = GridSearchCV(xgb1,

                        parameters,

                        cv = 2,

                        n_jobs = 5,

                        verbose=True)



xgb_grid.fit(X_train_val,

         y_train_val)



print(xgb_grid.best_score_)

print(xgb_grid.best_params_)
#XGBoost Score

predictionXgbrCV=xgb_grid.predict(X_test)

precision_score(y_test, predictionXgbrCV.round()), recall_score(y_test, predictionXgbrCV.round()), f1_score(y_test, predictionXgbrCV.round())
#Feature Importance with Random Forest

import eli5

from eli5.sklearn import PermutationImportance

perm=PermutationImportance(rand, random_state=1).fit(X_scaled,y)

eli5.show_weights(perm, feature_names=X_num.columns.tolist())
#Drop Less Importance Feature Columns

X_RF=X_numerical.drop(['Geography_Spain','Gender_Male','HasCrCard','Geography_France','Gender_Female'],axis='columns')

X_scaled_RF=scaler.fit_transform(X_RF)

X_train_val_RF, X_test_RF, y_train_val_RF, y_test_RF = train_test_split(X_scaled_RF, y, test_size=.2, random_state=50)

rand_F=RandomForestClassifier(n_estimators=300)

rand_F.fit(X_train_val_RF, y_train_val_RF)

predictionRand_F=rand_F.predict(X_test_RF)

precision_score(y_test_RF, predictionRand_F), recall_score(y_test_RF, predictionRand_F),f1_score(y_test_RF, predictionRand_F)
#Default threshold

thress=0.5

predictionRand_F = (rand_F.predict_proba(X_test_RF)[:,1] > thress)

print("Threshold of {:6.2f}:".format(thress))

print("Precision: {:6.4f},   Recall: {:6.4f},   F1: {:6.4f}".format(precision_score(y_test_RF, predictionRand_F), 

                                                     recall_score(y_test_RF, predictionRand_F),f1_score(y_test_RF, predictionRand_F)))
#Optimum threshold

thress=0.33

predictionRand_F = (rand_F.predict_proba(X_test_RF)[:,1] > thress)

print("Threshold of {:6.2f}:".format(thress))

print("Precision: {:6.4f},   Recall: {:6.4f},   F1: {:6.4f}".format(precision_score(y_test_RF, predictionRand_F), 

                                                     recall_score(y_test_RF, predictionRand_F),f1_score(y_test_RF, predictionRand_F)))
confusion_matrix(y_test_RF, predictionRand_F)