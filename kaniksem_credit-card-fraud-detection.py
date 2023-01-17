# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



credit_card=pd.read_csv('../input/creditcard.csv')

# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV



%matplotlib inline
sns.countplot(credit_card['Class'])
from sklearn.preprocessing import StandardScaler

# normalizing the amount column

scaler = StandardScaler()

scaler.fit(credit_card['Amount'].values.reshape(-1,1))

credit_card['normalized_amount'] = scaler.transform(credit_card['Amount'].values.reshape(-1,1))
# dropping the time column and adding the normalized amount column

X, y = credit_card[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',

       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',

       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'normalized_amount']], credit_card['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

predictions_logreg = log_reg.predict(X_test)

print(metrics.confusion_matrix(y_test,predictions_logreg))

print(metrics.classification_report(y_test,predictions_logreg))



# If I use Logistic Regression with default parameters on the original dataset, recall is only 57%
# In order to address imbalanced class, I am using the RandomOverSampler ; 

# RandomOverSampler generates new samples by randomly sampling with replacement the current available samples



from imblearn.over_sampling import RandomOverSampler



X_resampled, y_resampled = RandomOverSampler(random_state=0).fit_sample(X_train, y_train)
# Running a GridSearchCV on the re-sampled data to find the optimal value of C



param_grid_logreg = {'C':[1, 0.1, 0.01, 0.001, 0.0001, 0.00001]}

grid_logreg = GridSearchCV(LogisticRegression(), param_grid=param_grid_logreg, 

                           verbose=3, scoring='recall')

grid_logreg.fit(X_resampled, y_resampled)
grid_logreg.best_params_
# Running Logistic Regression on the re-sampled training data with an optimal value of C of 0.00001



log_reg_RandomSampler = LogisticRegression(C=0.00001)

log_reg_RandomSampler.fit(X_resampled, y_resampled)

predictions_logreg_RandomSampler = log_reg_RandomSampler.predict(X_test)

print(metrics.confusion_matrix(y_test, predictions_logreg_RandomSampler))

print(metrics.classification_report(y_test, predictions_logreg_RandomSampler))



# The recall accuracy now improves to 93% with re-sampled training data and an optimal value of C
svc = SVC()

svc.fit(X_train, y_train)

predictions_svc = svc.predict(X_test)

print(metrics.confusion_matrix(y_test,predictions_svc))

print(metrics.classification_report(y_test,predictions_svc))



# If I use SVC with default parameters on the original dataset, recall is only 61%
# Running RandomizedSearchCV for SVC to optimize RECALL using the following :

param_dist = {'C':[0.1,1,10,100,1000], 'gamma':[0.1, 0.001, 0.0001, 0.00001]}

grid = RandomizedSearchCV(SVC(),param_distributions=param_dist,verbose=5,n_iter=20, n_jobs=3, scoring='recall')

grid.fit(X_train, y_train)

# best parameters for SVC are  C=100, gamma=0.0001
grid.best_params_
model_SVC_RandomOverSampler = SVC(C=100, gamma=0.0001)

model_SVC_RandomOverSampler.fit(X_resampled, y_resampled)

predictions_SVC_RandomOverSampler = model_SVC_RandomOverSampler.predict(X_test)
print(metrics.confusion_matrix(y_test,predictions_SVC_RandomOverSampler))

print(metrics.classification_report(y_test,predictions_SVC_RandomOverSampler))

# The recall accuracy now improves to 93% with re-sampled training data and an optimal value of C and gamma
rf = RandomForestClassifier()

rf.fit(X_train, y_train)

predictions_rf = rf.predict(X_test)

print(metrics.confusion_matrix(y_test,predictions_rf))

print(metrics.classification_report(y_test,predictions_rf))



# If I use Random Forest with default parameters on the original dataset, recall is only 74%
param_dist_rf = {'max_depth':[3,5,7,9], 'max_features':[5,7,10,12], 'n_estimators':[10,15,20,25]} 

grid_rf = RandomizedSearchCV(RandomForestClassifier(),param_distributions=param_dist_rf,

                             verbose=5,n_iter=20, n_jobs=3, scoring='recall')

grid_rf.fit(X_train,y_train)



#  after running RandomizedSearchCV, best parameters for Random Forest are 'n_estimators': 10, 

# 'max_features': 10, 'max_depth': 7
model_rf_randomSampler = RandomForestClassifier(n_estimators=10, max_features=10, max_depth=7)

model_rf_randomSampler.fit(X_resampled, y_resampled)

predictions_rf_randomSampler = model_rf_randomSampler.predict(X_test)
print(metrics.confusion_matrix(y_test, predictions_rf_randomSampler))

print(metrics.classification_report(y_test, predictions_rf_randomSampler))



# The recall accuracy now improves to 88% with re-sampled training data 

# and an optimal value of n_estimators, max_features, max_depth 