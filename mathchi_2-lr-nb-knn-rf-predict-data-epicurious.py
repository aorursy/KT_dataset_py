# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statsmodels.api as sm

import statsmodels.formula.api as smf

import seaborn as sns

from sklearn.preprocessing import scale 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score,roc_curve

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier



from warnings import filterwarnings

filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
epicurious = pd.read_csv("../input/epirecipes/epi_r.csv")
epicurious.head(2)
epicurious.info(5)
epicurious.describe().T
epicurious = epicurious[epicurious['calories'] < 10000].dropna()
sns.set(style="darkgrid")

g = sns.regplot(x="calories", y="dessert", data=epicurious, fit_reg=False)

g.figure.set_size_inches(5, 5)



epicurious = epicurious[:][:500]      # lets take limit for speed regression calculating
# "title" feature we dont need and lets drop it

epicurious.drop("title", inplace = True, axis=1)



y = epicurious.dessert.values

X = epicurious.drop(["dessert"], axis = 1)
# see how many null values we have then we dont need to normalize



epicurious['dessert'].isnull().sum()
loj = LogisticRegression(solver = "liblinear")

loj_model = loj.fit(X,y)

loj_model
loj_model.intercept_      # constant value

loj_model.coef_           # independent values
y_pred = loj_model.predict(X)        # predict

confusion_matrix(y, y_pred)          # confussion matrix
accuracy_score(y, y_pred)
print(classification_report(y, y_pred))
# Model predict

loj_model.predict(X)[0:20]
loj_model.predict_proba(X)[0:10][:,0:2]
# Now lets try model 'predict_proba' probability



y_probs = loj_model.predict_proba(X)

y_probs = y_probs[:,1]

y_probs[0:20]
# giving limit for values



y_pred = [1 if i > 0.5 else 0 for i in y_probs]
# and compare with above you can see what happened

y_pred[0:20]
confusion_matrix(y, y_pred)
accuracy_score(y, y_pred)
print(classification_report(y, y_pred))
logit_roc_auc = roc_auc_score(y, loj_model.predict(X))



fpr, tpr, thresholds = roc_curve(y, loj_model.predict_proba(X)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

plt.show()



# blue line: which we set our model

# red line: if we dont do it what can we take result
# lets split test train set



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size = 0.30, 

                                                    random_state = 42)
# set model



loj = LogisticRegression(solver = "liblinear")

loj_model = loj.fit(X_train,y_train)

loj_model
accuracy_score(y_test, loj_model.predict(X_test))
# with cross validation 



cross_val_score(loj_model, X_test, y_test, cv = 10).mean()
nb = GaussianNB()

nb_model = nb.fit(X_train, y_train)

nb_model
nb_model.predict(X_test)[0:20]
nb_model.predict_proba(X_test)[0:10]
# predict

y_pred = nb_model.predict(X_test)

accuracy_score(y_test, y_pred)
cross_val_score(nb_model, X_test, y_test, cv = 10).mean()
knn = KNeighborsClassifier()

knn_model = knn.fit(X_train, y_train)

knn_model
y_pred = knn_model.predict(X_test)

accuracy_score(y_test, y_pred)
# get detail print



print(classification_report(y_test, y_pred))
# find KNN parameters

knn_params = {"n_neighbors": np.arange(1,50)}
# fit model classification & CV



knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, knn_params, cv=10)

knn_cv.fit(X_train, y_train)
# this is only observation



print("Best score:" + str(knn_cv.best_score_))

print("Best parameters: " + str(knn_cv.best_params_))
knn = KNeighborsClassifier(4)

knn_tuned = knn.fit(X_train, y_train)

knn_tuned.score(X_test, y_test)
y_pred = knn_tuned.predict(X_test)

accuracy_score(y_test, y_pred)
rf_model = RandomForestClassifier().fit(X_train, y_train)

rf_model
y_pred = rf_model.predict(X_test)

accuracy_score(y_test, y_pred)
rf_params = {"max_depth": [2,5,8,10],

            "max_features": [2,5,8],

            "n_estimators": [10,500,1000],

            "min_samples_split": [2,5,10]}
rf_model = RandomForestClassifier()



rf_cv_model = GridSearchCV(rf_model, 

                           rf_params, 

                           cv = 10, 

                           n_jobs = -1, 

                           verbose = 2) 
rf_cv_model.fit(X_train, y_train)
print("Best Parameters: " + str(rf_cv_model.best_params_))
# using given parameters then create final model



rf_tuned = RandomForestClassifier(max_depth = 8, 

                                  max_features = 8, 

                                  min_samples_split = 5,

                                  n_estimators = 10)



rf_tuned.fit(X_train, y_train)
# tunned test model predict accuracy score



y_pred = rf_tuned.predict(X_test)

accuracy_score(y_test, y_pred)