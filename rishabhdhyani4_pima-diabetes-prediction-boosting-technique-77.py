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
# Plotting Libraries



import matplotlib.pyplot as plt

import seaborn as sns

import cufflinks as cf

%matplotlib inline



# Metrics for Classification technique



from sklearn.metrics import classification_report,confusion_matrix,accuracy_score



# Scaler



from sklearn.preprocessing import RobustScaler, StandardScaler



# Cross Validation



from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split



# Linear Models



from sklearn.linear_model import LogisticRegression



# Ensemble Technique



from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier



# Other model



from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC



# Model Stacking 



from mlxtend.classifier import StackingCVClassifier



# Other libraries



from datetime import datetime

from scipy.stats import skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.impute import SimpleImputer

from numpy import nan

from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform
data = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

data.head(6) # Mention no of rows to be displayed from the top in the argument
# Shape of the dataset



data.shape
data.info()
data.describe().transpose()
plt.figure(figsize=(20,12))

sns.set_context('notebook',font_scale = 1.3)

sns.heatmap(data.corr(),annot=True,cmap='coolwarm')

plt.tight_layout()
sns.countplot(x=data['Outcome'],data = data)
X = data.drop('Outcome',axis = 1)

y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)
print("total number of rows : {0}".format(len(data)))

print("number of rows missing glucose_conc: {0}".format(len(data.loc[data['Glucose'] == 0])))

print("number of rows missing diastolic_bp: {0}".format(len(data.loc[data['BloodPressure'] == 0])))

print("number of rows missing insulin: {0}".format(len(data.loc[data['Insulin'] == 0])))

print("number of rows missing bmi: {0}".format(len(data.loc[data['BMI'] == 0])))

print("number of rows missing diab_pred: {0}".format(len(data.loc[data['DiabetesPedigreeFunction'] == 0])))

print("number of rows missing age: {0}".format(len(data.loc[data['Age'] == 0])))
# Filling Zero values



fill_values = SimpleImputer(missing_values=0, strategy="mean")



X_train = fill_values.fit_transform(X_train)

X_test = fill_values.fit_transform(X_test)
# RandomForestClassifier



random_forest_model = RandomForestClassifier(random_state = 42)



random_forest_model.fit(X_train, y_train.ravel())
predict_train_data = random_forest_model.predict(X_test)



print("Accuracy = {0:.3f}".format(accuracy_score(y_test, predict_train_data)))
## Hyperparameter Optimzation



params1={

    

    "n_estimators" : [100, 300, 500, 800, 1200], 

    "max_depth" : [5, 8, 15, 25, 30],

    "min_samples_split" : [2, 5, 10, 15, 100],

    "min_samples_leaf" : [1, 2, 5, 10] 



}
rfm = RandomForestClassifier(random_state = 42)
rfms = RandomizedSearchCV(rfm,param_distributions=params1,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
# Here we go

start_time = timer(None) # timing starts from this point for "start_time" variable

rfms.fit(X_train,y_train.ravel())

timer(start_time) # timing ends here for "start_time" variable
rfms.best_estimator_
model1 = RandomForestClassifier(max_depth=8, min_samples_split=10, n_estimators=500,

                       random_state=42)
model1.fit(X_train,y_train)

y_pred1 = model1.predict(X_test)
print(accuracy_score(y_test,y_pred1))

print(confusion_matrix(y_test,y_pred1))
model2 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

              importance_type='gain', interaction_constraints='',

              learning_rate=0.300000012, max_delta_step=0, max_depth=6,

              min_child_weight=1, missing=nan, monotone_constraints='()',

              n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,

              tree_method='exact', validate_parameters=1, verbosity=None)
model2.fit(X_train,y_train.ravel())

y_pred2 = model2.predict(X_test)
print(accuracy_score(y_test,y_pred2))

print(confusion_matrix(y_test,y_pred2))
model3 = CatBoostClassifier()
model3.fit(X_train,y_train)

y_pred3 = model3.predict(X_test)
print(accuracy_score(y_test,y_pred3))

print(confusion_matrix(y_test,y_pred3))
params4 = {'C': [0.1, 1, 10, 100, 1000],  

              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 

              'kernel': ['rbf']} 
svcs = RandomizedSearchCV(SVC(),param_distributions=params4,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
# Here we go

start_time = timer(None) # timing starts from this point for "start_time" variable

svcs.fit(X_train,y_train.ravel())

timer(start_time) # timing ends here for "start_time" variable
svcs.best_estimator_
model4 = SVC(C=0.1, gamma=0.001)
model4.fit(X_train,y_train)

y_pred4 = model4.predict(X_test)
print(accuracy_score(y_test,y_pred4))

print(confusion_matrix(y_test,y_pred4))
params5 = {'n_estimators':[500,1000,2000],'learning_rate':[.001,0.01,.1]}
adas = RandomizedSearchCV(AdaBoostClassifier(),param_distributions=params5,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
# Here we go

start_time = timer(None) # timing starts from this point for "start_time" variable

adas.fit(X_train,y_train.ravel())

timer(start_time) # timing ends here for "start_time" variable
adas.best_estimator_
model5 = AdaBoostClassifier(learning_rate=0.01, n_estimators=500)
model5.fit(X_train,y_train)

y_pred5 = model5.predict(X_test)
print(accuracy_score(y_test,y_pred5))

print(confusion_matrix(y_test,y_pred5))
params6 = {

    'learning_rate': [ 0.1],

    'num_leaves': [31],

    'boosting_type' : ['gbdt'],

    'objective' : ['binary']

}
lgbs = RandomizedSearchCV(LGBMClassifier(),param_distributions=params6,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
# Here we go

start_time = timer(None) # timing starts from this point for "start_time" variable

lgbs.fit(X_train,y_train.ravel())

timer(start_time) # timing ends here for "start_time" variable
lgbs.best_estimator_
model6 = LGBMClassifier(objective='binary')
model6.fit(X_train,y_train)

y_pred6 = model6.predict(X_test)
print(accuracy_score(y_test,y_pred6))

print(confusion_matrix(y_test,y_pred6))
model7 = GradientBoostingClassifier(random_state = 42)
model7.fit(X_train,y_train)

y_pred7 = model7.predict(X_test)
print(accuracy_score(y_test,y_pred7))

print(confusion_matrix(y_test,y_pred7))
## Stacking of Models



model8 = StackingCVClassifier(classifiers=[model1,model2,model3,model5,model6,model7],

                            meta_classifier=model1,

                            random_state=42)
model8.fit(X_train,y_train.ravel())
y_pred8 = model8.predict(X_test)
print(accuracy_score(y_test,y_pred8))

print(confusion_matrix(y_test,y_pred8))