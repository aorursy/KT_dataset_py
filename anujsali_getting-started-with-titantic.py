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
## 1. GET THE DATA
train_set=pd.read_csv('/kaggle/input/titanic/train.csv')
train_set
train_set.head()
train_set.info()
train_set.describe()
## 2. Gaining Insights:
import matplotlib.pyplot as plt
%matplotlib inline
train_set.hist(bins=50,figsize=(20,15))
import seaborn as sns
sns.pairplot(train_set)
## 3.Exploratory Data Analysis:
#Since, training data is not huge(in millions), its not advisable to use train_test_split().Lets use StratifiedShuffleSplit() for dividing 

#training set into train and test sets.
# On second thought, lets not use it :p
## 4. Data Preparation For ML Algorithms
features=train_set.drop(['Survived'],axis=1)
label=train_set['Survived']
features
label
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
num_pipeline=Pipeline([('imputer',SimpleImputer(strategy='median')),

                      ('standard_scaler',StandardScaler())])
num_attribs=features.columns.values[[0,1,4,5,6,8]]
num_attribs
cat_attribs=features.columns.values[[3]]
cat_attribs
full_pipeline=ColumnTransformer([('num_pipeline',num_pipeline,num_attribs),

                                ('onehotencoder',OneHotEncoder(),cat_attribs)])
features_prepared=full_pipeline.fit_transform(features)
features_prepared
features_prepared.shape
## Train Different Classifiers:
from sklearn.linear_model import SGDClassifier
sgd_clf=SGDClassifier()
sgd_clf.fit(features_prepared,label)
predicted_sgd=sgd_clf.predict(features_prepared)
predicted_sgd
label
from sklearn.model_selection import cross_val_score
scores=cross_val_score(sgd_clf,features_prepared,label,cv=3,scoring='accuracy')
scores
from sklearn.ensemble import RandomForestClassifier

rf_clf=RandomForestClassifier(random_state=42)

rf_clf.fit(features_prepared,label)
predicted_rf=rf_clf.predict(features_prepared)
scores_rf=cross_val_score(rf_clf,features_prepared,label,cv=3,scoring='accuracy')

scores_rf
print('Mean:\t ',scores_rf.mean())

print('Standard Deviation:\t ',scores_rf.std())
## Fine Tuning Our Best Model:
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from scipy.stats import randint
param_dist={'n_estimators':randint(low=1,high=100),'max_features':randint(low=1,high=3)}

rand_search=RandomizedSearchCV(rf_clf,param_distributions=param_dist,scoring='accuracy',cv=3,random_state=42,n_iter=100)

rand_search.fit(features_prepared,label)
rand_predicted=rand_search.predict(features_prepared)
rand_predicted
rand_search.best_params_
rand_search.best_estimator_
cvres=rand_search.cv_results_
cvres
pd.DataFrame(cvres)
for best_score,params in zip(cvres['mean_test_score'],cvres['params']):

    print(best_score,params)
grid_params=[{'bootstrap':[True],'n_estimators':[48,51,52,53,79,81,83,86,88,90],'max_features':[1,2]}]

grid_search=GridSearchCV(rf_clf,grid_params,cv=3,scoring='accuracy',return_train_score=True)

grid_search.fit(features_prepared,label)

predicted_grid=grid_search.predict(features_prepared)

predicted_grid
predicted_grid=grid_search.predict(features_prepared)

predicted_grid
grid_search.best_params_
cvres1=grid_search.cv_results_

pd.DataFrame(cvres1)
for best_score,params in zip(cvres1['mean_test_score'],cvres1['params']):

    print(best_score,params)
## Testing Our Fined Tuned Best Model On Test set:
final_model=grid_search.best_estimator_
test_set=pd.read_csv('/kaggle/input/titanic/test.csv')
final_features=full_pipeline.transform(test_set)
final_predictions=final_model.predict(final_features)
final_predictions
final_scores=cross_val_score(final_model,final_features,final_predictions,cv=3,scoring='accuracy')

final_scores.mean()
final=pd.DataFrame(final_predictions,columns=['Survived'])
final['PassengerId']=test_set['PassengerId']
cols=final.columns.tolist()
cols
cols = cols[-1:] +cols[:-1]
cols
final=final[cols]
final
final.to_csv('Final_Submission.csv',index=False)
##Lets submit the final submission now!!!