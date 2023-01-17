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
# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)



# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"



# import usual libraries 

import pandas as pd

import numpy as np



# visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# consistent sized plots 

from pylab import rcParams

rcParams['figure.figsize'] = 12,5

rcParams['axes.labelsize']= 14

rcParams['xtick.labelsize'] =12

rcParams['ytick.labelsize'] = 12 



# handle operating system dependencies 

import os



# handle unwanted warnings 

import warnings

warnings.filterwarnings(action='ignore',message='^internal gelsd')
# READ THE TRAINING DATA AND TEST DATA INTO DATAFRAMES

train_credit = pd.read_csv('../input/south-german-credit-prediction/train.csv')

test_credit = pd.read_csv('../input/south-german-credit-prediction/test.csv') 
train_credit.head(20)
train_credit.columns
# assign the corresponding english column names

train_credit.columns = ['Id','status', 'duration', 'credit_history', 'purpose', 'amount', 'savings', 'employment_duration', 'installment_rate', 'personal_status_sex', 'other_debtors', 'present_residence', 'property', 'age', 'other_installment_plans', 'housing', 'number_credits', 'job', 'people_liable', 'telephone', 'foreign_worker', 'credit_risk']

test_credit.columns =  ['Id','status', 'duration', 'credit_history', 'purpose', 'amount', 'savings', 'employment_duration', 'installment_rate', 'personal_status_sex', 'other_debtors', 'present_residence', 'property', 'age', 'other_installment_plans', 'housing', 'number_credits', 'job', 'people_liable', 'telephone', 'foreign_worker']
# check the dataframe after rename of the columns

train_credit.head(5)
train_credit.describe().transpose()
# check info 

train_credit.info()
# explicit check for any null values in the dataframe 

train_credit.isnull().sum()
sns.countplot(train_credit['credit_risk'])
sns.countplot(train_credit['foreign_worker'],hue=train_credit['credit_risk'])
sns.countplot(train_credit['credit_risk'],hue=train_credit['purpose'])
plt.hist(train_credit['amount'],bins=30);
# check the bad loans

train_credit[train_credit['credit_risk']==0]
plt.hist(train_credit[train_credit['credit_risk']==0]['amount'])

plt.title('Bad Loans Amount Histogram')
# check the good and bad loan risk 

train_credit['credit_risk'].value_counts()
data= train_credit.copy()
# drop the Id column as it is not useful for the model 

data.drop(['Id'],inplace=True,axis=1)
data.head(3)
log_amount = np.log(data['amount'])

sns.distplot(log_amount,bins=20)
# import the varios classifier models

from sklearn.linear_model import SGDClassifier,LogisticRegression

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_predict, cross_val_score,KFold, RepeatedStratifiedKFold
new_data = train_credit.copy()
new_data.info()
new_data.drop('Id',axis=1,inplace=True)
new_data['log_amount'] = round(np.log(new_data['amount']),2)
new_data.drop('amount',axis=1,inplace=True)
new_data['log_age'] =  round(np.log(new_data['age']),2)
new_data['log_duration'] = round(np.log(new_data['duration']),2)
new_data.drop(['age','duration'],axis=1,inplace=True)
new_data.head()
new_data.tail()
X_full = new_data.drop('credit_risk',axis=1)

y_full = new_data['credit_risk']
# this will ensure that the data is randomized and then split into train and test 

# alternatively StratifiedRandomSplit is also recommended

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
# try various models and pick the best one for further tuning 

def cross_validate(X = X_train,y = y_train):

    

    warnings.filterwarnings(action='ignore',message='')



    models = []

    models.append(('RF',RandomForestClassifier()))

    models.append(('GB',GradientBoostingClassifier()))

    models.append(('SVC',SVC()))

    models.append(('SGD',SGDClassifier()))

    models.append(('LogReg',LogisticRegression()))

    models.append(('AdaBoost',AdaBoostClassifier()))

    models.append(('Bag',BaggingClassifier()))

    models.append(('xgboost',XGBClassifier()))

    models.append(('lightgbm',LGBMClassifier()))

    models.append(('Dtree',DecisionTreeClassifier()))

    



    results = []

    names = []

    scoring ='accuracy'



    for name,model in models:

        #kfold = KFold(n_splits=10,random_state=42)

        kfold = RepeatedStratifiedKFold(n_splits=10,random_state=42,n_repeats=3)

        cv_results = cross_val_score(model,X,y,cv=kfold,scoring=scoring)

        results.append(cv_results)

        names.append(name)

        print (f'Model:{name},Mean: {cv_results.mean()},Std Dev: {cv_results.std()}')
cross_validate(X_train,y_train)
test_sub = test_credit.copy()
test_sub.drop('Id',axis=1,inplace=True)
test_sub['log_amount'] = round(np.log(test_sub['amount']),2)
test_sub['log_age'] =  round(np.log(test_sub['age']),2)
test_sub['log_duration'] = round(np.log(test_sub['duration']),2)
test_sub.drop(['amount','age','duration'],axis=1,inplace=True)
test_sub.head()
from imblearn.over_sampling import ADASYN

from imblearn.over_sampling import SMOTE # use either ADASYN or SMOTE

from collections import Counter
ada = ADASYN(sampling_strategy='minority',random_state=42,n_neighbors=7)

X_res,y_res = ada.fit_resample(X_train,y_train)

Counter(y_res)
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
cross_validate(X_res,y_res)
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
param_grid = [{'n_estimators': [3, 10, 30], 'max_depth': [2, 4, 6, 8],'booster': ['gbtree','dart'],

              'learning_rate':[0.3,0.5,0.01,0.1]}]
xgb_clf = XGBClassifier(random_state=42)



grid_search = GridSearchCV(xgb_clf, param_grid=param_grid, cv=5,

                           scoring='accuracy',

                           return_train_score=True)

grid_search.fit(X_res,y_res)
grid_search.best_params_
from scipy.stats import randint



param_distribs = {

        'n_estimators': randint(low=1, high=500),

        'max_depth': randint(low=1, high=10),

        'max_features':randint(low=1,high=10),

        

    }



rf_clf = RandomForestClassifier(random_state=42)

rnd_search = RandomizedSearchCV(rf_clf, param_distributions=param_distribs,

                                n_iter=10, cv=5, scoring='accuracy', random_state=42)

rnd_search.fit(X_res,y_res)
rnd_search.best_params_
rf_clf = RandomForestClassifier(random_state=42,max_depth=8,max_features=6,n_estimators=386)
# hyper parameters selcted based on grid search 

xgb_clf =  XGBClassifier(n_estimators=30,max_depth=8,random_state=42,learning_rate=0.3,

                        booster='gbtree')
svc_clf = SVC(random_state=42)   # with default paramters
gb_clf = GradientBoostingClassifier(random_state=42) # default parameters
bag_clf = BaggingClassifier(random_state=42,base_estimator=XGBClassifier())
xgb_clf.fit(X_res,y_res)
rf_clf.fit(X_res,y_res)
svc_clf.fit(X_res,y_res)
gb_clf.fit(X_res,y_res)
bag_clf.fit(X_res,y_res)
predictions_train_xgb = xgb_clf.predict(X_test)
predictions_train_rf = rf_clf.predict(X_test)
predictions_train_svc = svc_clf.predict(X_test)
predictions_train_gb = gb_clf.predict(X_test)
predictions_train_bag = bag_clf.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score

print('Accuracy XGBoost...{}'.format(accuracy_score(y_test,predictions_train_xgb)))

print('Accuracy RForest...{}'.format(accuracy_score(y_test,predictions_train_rf)))

print('Accuracy SupportVector...{}'.format(accuracy_score(y_test,predictions_train_svc)))

print('Accuracy GBoost...{}'.format(accuracy_score(y_test,predictions_train_gb)))

print('Accuracy Bagging...{}'.format(accuracy_score(y_test,predictions_train_gb)))
print('Precision XGBoost...{}'.format(precision_score(y_test,predictions_train_xgb)))

print('Precision RForest...{}'.format(precision_score(y_test,predictions_train_rf)))

print('Precision SupportVector...{}'.format(precision_score(y_test,predictions_train_svc)))

print('Precision GBoost...{}'.format(precision_score(y_test,predictions_train_gb)))

print('Precision Bagging...{}'.format(precision_score(y_test,predictions_train_gb)))
print('Recall XGBoost...{}'.format(recall_score(y_test,predictions_train_xgb)))

print('Recall RForest...{}'.format(recall_score(y_test,predictions_train_rf)))

print('Recall SupportVector...{}'.format(recall_score(y_test,predictions_train_svc)))

print('Recall GBoost...{}'.format(recall_score(y_test,predictions_train_gb)))

print('Recall Bagging...{}'.format(recall_score(y_test,predictions_train_gb)))
print('XGBoost_Confusion Matrix')

print(confusion_matrix(y_test,predictions_train_xgb))

print('RandomForest_Confusion Matrix')

print(confusion_matrix(y_test,predictions_train_rf))

print('SupportVector_Confusion Matrix')

print(confusion_matrix(y_test,predictions_train_svc))

print('GradientBoosting_Confusion Matrix')

print(confusion_matrix(y_test,predictions_train_gb))

print('Bagging_Confusion Matrix')

print(confusion_matrix(y_test,predictions_train_gb))
train_oversample = pd.concat([X_res,X_test],axis=0)
test_oversample = pd.concat([y_res,y_test],axis=0)
train_oversample.shape
train_oversample.columns
test_oversample.shape # contains 0 and 1 for the credit risk
train_oversample = scalar.fit_transform(train_oversample)
xgb_clf.fit(train_oversample,test_oversample)
rf_clf.fit(train_oversample,test_oversample)
svc_clf.fit(train_oversample,test_oversample)
gb_clf.fit(train_oversample,test_oversample)
bag_clf.fit(train_oversample,test_oversample)
predictions_final_xgb = xgb_clf.predict(scalar.transform(test_sub))

predictions_final_rf = rf_clf.predict(scalar.transform(test_sub))

predictions_final_svc = svc_clf.predict(scalar.transform(test_sub))

predictions_final_gb = gb_clf.predict(scalar.transform(test_sub))

predictions_final_bag = bag_clf.predict(scalar.transform(test_sub))
s_xgb = pd.Series(predictions_final_xgb, name='XGB')

s_rf = pd.Series(predictions_final_rf, name='RF')

s_svc = pd.Series(predictions_final_svc, name='SVC')

s_gb = pd.Series(predictions_final_gb, name='GB')

s_bag = pd.Series(predictions_final_bag, name='BAG')

idx = test_credit['Id']
model_pred = pd.concat([idx,s_xgb,s_rf,s_svc,s_gb,s_bag],axis=1)

model_pred.head()
model_pred['vote'] = model_pred[['XGB','RF','SVC','GB','BAG']].sum(axis=1)
model_pred.head()
# criteria to select the final credit risk score 

def vote(vote_sum):

    if vote_sum >=2:

        return 1

    else:

        return 0
model_pred['kredit'] = model_pred['vote'].apply(vote)
model_pred.tail()
submission = model_pred.drop(['XGB','RF','SVC','GB','BAG','vote'],axis=1)
submission.tail(20)
submission.to_csv('AggregatedModel_Predictions.csv',index=False)
scalar = StandardScaler()

X_res_sc =  scalar.fit_transform(X_res)

X_test_sc = scalar.transform(X_test)
X_train_ar = np.asarray(X_res_sc)

y_train_ar = np.asarray(y_res)

X_test_ar = np.asarray(X_test_sc)

y_test_ar = np.asarray(y_test)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Flatten,Dropout

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow import keras
keras.backend.clear_session()

model_ann = Sequential()



# add 3 dense layers and the final output layer

model_ann.add(Flatten())

model_ann.add(Dense(units=300)) # activations relu,tanh,elu all resulted in exploding val_loss

model_ann.add(Dense(units=200))

model_ann.add(Dense(units=100))

model_ann.add(Dropout(0.5))

model_ann.add(Dense(units=10))





# final output layer

model_ann.add(Dense(units=1,activation='sigmoid'))



# compile the model

model_ann.compile(optimizer='nadam',metrics=['accuracy'],loss='binary_crossentropy')
early_stop = EarlyStopping(patience=50,monitor='val_loss',restore_best_weights=True)
model_ann.fit(X_train_ar,y_train_ar,epochs=300,callbacks=[early_stop],

             validation_data=(X_test_ar,y_test_ar))
predictions_ann = model_ann.predict_classes(X_test_ar)
print(accuracy_score(y_test,predictions_ann))