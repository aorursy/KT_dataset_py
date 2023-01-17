# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as sp

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

import warnings

import warnings

warnings.filterwarnings('ignore')

from IPython.display import HTML
df = pd.read_csv('../input/winequality-red.csv')
df.head()
df.info()
df.describe()
sns.pairplot(df, height=2.5);
corr = df.corr()

plt.figure(figsize = (16,10))

sns.heatmap(corr,annot=True, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()
X = df.iloc[:,:-1]

y = df.iloc[:,-1]
plt.hist(y,bins=len(set(y)))

plt.title('Distribution of target values')

plt.xlabel('Classes')

plt.ylabel('count')

plt.show()
print(y.value_counts())
random_state=10

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=random_state)
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
GBM1 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000,max_depth=3, min_samples_split=2, min_samples_leaf=1)

GBM1.fit(X_train,y_train)

predictors=list(X_train)

feat_imp = pd.Series(GBM1.feature_importances_, predictors).sort_values(ascending=False)

feat_imp.plot(kind='bar', title='Importance of Features')

plt.ylabel('Feature Importance Score')

GBM1_score = GBM1.score(X_test, y_test)

print('Accuracy of the GBM on test set: {:.3f}'.format(GBM1_score))

GBM1_pred=GBM1.predict(X_test)

print(classification_report(y_test, GBM1_pred))
from sklearn.metrics import confusion_matrix

CM = confusion_matrix(y_test, GBM1_pred)



z= list(set(y))

sns.heatmap(CM, annot=True, fmt="d", xticklabels=z, yticklabels=z )

plt.title('confusion matrix - GBM model without tuning')

plt.xlabel('Real')

plt.ylabel('Predicted')

plt.show()
df_sub_set=df.loc[(df['quality']!=3) &(df['quality']!=8)]
X_train_sub_set, X_test_sub_set, y_train_sub_set, y_test_sub_set = train_test_split(df_sub_set.iloc[:,:-1],df_sub_set.iloc[:,-1], test_size=0.2,random_state=random_state)
GBM1_sub_set = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000,max_depth=3, min_samples_split=2, min_samples_leaf=1)

GBM1_sub_set.fit(X_train_sub_set,y_train_sub_set)

predictors_sub_set=list(X_train_sub_set)

feat_imp_sub_set = pd.Series(GBM1.feature_importances_, predictors_sub_set).sort_values(ascending=False)

feat_imp_sub_set.plot(kind='bar', title='Importance of Features')

plt.ylabel('Feature Importance Score')

GBM1_score_sub_set = GBM1_sub_set.score(X_test_sub_set, y_test_sub_set)

print('Accuracy of the GBM on test set: {:.3f}'.format(GBM1_score_sub_set))

GBM1_pred_sub_set=GBM1_sub_set.predict(X_test_sub_set)

print(classification_report(y_test_sub_set, GBM1_pred_sub_set))
tuning_n_estimators_prams = {'learning_rate':[0.1,0.05,0.01], 'n_estimators':[500,1000,1500,2000]}

GBC =GradientBoostingClassifier(random_state=random_state)

tuning_n_estimators = GridSearchCV(estimator=GBC,param_grid = tuning_n_estimators_prams, scoring='accuracy', cv=5)

tuning_n_estimators.fit(X_train,y_train)

print("Best score based on {} is {}".format(tuning_n_estimators.best_params_,tuning_n_estimators.best_score_))
tuning_max_depth_prams = {'max_depth':[3,5,7] }

tuning_max_depth = GridSearchCV(estimator =GradientBoostingClassifier(learning_rate=0.1,n_estimators=500), 

            param_grid = tuning_max_depth_prams, scoring='accuracy', cv=5)

tuning_max_depth.fit(X_train,y_train)

print("Best score based on {} is {}".format(tuning_max_depth.best_params_,tuning_max_depth.best_score_))
tuning_min_samples_prams = {'min_samples_split':[2,6,10], 'min_samples_leaf':[1,3,5]}



tuning_min_samples = GridSearchCV(estimator =GradientBoostingClassifier(learning_rate=0.1, n_estimators=500,max_depth=7), 

            param_grid = tuning_min_samples_prams, scoring='accuracy', cv=5)

tuning_min_samples.fit(X_train,y_train)

print("Best score based on {} is {}".format(tuning_min_samples.best_params_,tuning_min_samples.best_score_))
GBM_TUN = GradientBoostingClassifier(learning_rate=0.1, n_estimators=500,max_depth=7, min_samples_split=6, min_samples_leaf=5)

GBM_TUN.fit(X_train,y_train)

predictors_TUN = list(X_train)

feat_imp = pd.Series(GBM_TUN.feature_importances_, predictors_TUN).sort_values(ascending=False)

feat_imp.plot(kind='bar', title='Importance of Features')

plt.ylabel('Feature Importance Score')

print('Accuracy of the GBM on test set: {:.3f}'.format(GBM_TUN.score(X_test, y_test)))

GBM_TUN_pred=GBM_TUN.predict(X_test)

print(classification_report(y_test, GBM_TUN_pred))
CM_GBM_TUN = confusion_matrix(y_test, GBM_TUN_pred)

z= list(set(y))

sns.heatmap(CM_GBM_TUN, annot=True, fmt="d", xticklabels=z, yticklabels=z )

plt.title('confusion matrix - tuned GBM')

plt.xlabel('Real')

plt.ylabel('Predicted')

plt.show()
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold



GBC_CV = GradientBoostingClassifier(learning_rate=0.1, n_estimators=500,max_depth=7, min_samples_split=6, min_samples_leaf=5, random_state=random_state)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.00001,random_state=random_state)

kfold = KFold(n_splits=5, random_state=random_state)

scores = cross_val_score (GBC_CV, X_train, y_train, cv=kfold)

print("Scores on each subset:")

print(scores) 

avg = (100*np.mean(scores), 100*np.std(scores)/np.sqrt(scores.shape[0]))

print ("Average score and uncertainty: (%.2f +- %.3f)%%"%avg)
from sklearn.ensemble import RandomForestClassifier

np.random.seed(100)
df2 = pd.read_csv('../input/winequality-red.csv')
X2= df2.iloc[:,:-1]

y2 =df2.iloc[:,-1]

x_train2, x_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2,random_state=random_state)
rfc=RandomForestClassifier(random_state=random_state)



param_grid = {'n_estimators': [10, 50, 100,200]}



CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)

CV_rfc.fit(x_train2, y_train2)



print("Best params {}".format(CV_rfc.best_params_))
rfc_200 = RandomForestClassifier( n_estimators=200,random_state=random_state)

rfc_200.fit(x_train2, y_train2)

rfc_200_pred = rfc_200.predict(x_test2)

print('Accuracy of the RF on test set: {:.3f}'.format(rfc_200.score(x_test2, y_test2)))
cm_rfc = confusion_matrix(y_test2, rfc_200_pred)

z= (3,4,5,6,7,8,9)

sns.heatmap(cm_rfc, annot=True, fmt="d", xticklabels=z, yticklabels=z )

plt.title('confusion matrix - Random Forest')

plt.xlabel('Real')

plt.ylabel('Predicted')

plt.show()
X = df.iloc[:,:-1].values

y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.000001,random_state=10)

kfold = KFold(n_splits=5, random_state=10)

model = RandomForestClassifier(n_estimators=200)

results = cross_val_score(model, X_train, y_train, cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))