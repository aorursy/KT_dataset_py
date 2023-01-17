#Import relevant modules

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 

#read the data and take a look at the first few lines
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
train_data.head(20)

missing_vals_train = (train_data.isnull().sum())
print(missing_vals_train[missing_vals_train > 0])
missing_vals_test = (test_data.isnull().sum())
print(missing_vals_test[missing_vals_test > 0])
Columns_to_drop = ['Name','Ticket','Cabin','PassengerId']
reduced_data = train_data.drop(Columns_to_drop, axis=1)
reduced_data.head()
OHE_training_data = pd.get_dummies(reduced_data)
OHE_training_data.head(6)
y_training = OHE_training_data.Survived
X_training = OHE_training_data.drop('Survived',axis=1)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from xgboost import XGBClassifier

XGB_Pipeline = make_pipeline(Imputer(), XGBClassifier(random_state=1))
from sklearn.model_selection import cross_val_score
scores = cross_val_score(XGB_Pipeline, X_training, y_training, scoring='accuracy', cv=5)
print("mean score over 5 folds is %2f" %(scores.mean()))
#Import the models and then construct a pipeline for each including imputation
from sklearn.linear_model import LogisticRegression
LR_Pipeline = make_pipeline(Imputer(), LogisticRegression(random_state=1))
from sklearn.svm import SVC
SVC_Pipeline = make_pipeline(Imputer(), SVC(random_state=1))
from sklearn.ensemble import RandomForestClassifier
RF_Pipeline = make_pipeline(Imputer(), RandomForestClassifier(random_state=1))
from xgboost import XGBClassifier
XGBC_Pipeline = make_pipeline(Imputer(), XGBClassifier(random_state=1))

#Function that takes a model and returns its cross-validation score
def cv_score(model):
    scores = cross_val_score(model, X_training, y_training, scoring='accuracy', cv=5)
    return scores.mean()

#Calculate cross-validation scores using this function
CV_scores = pd.DataFrame({'Cross-validation score':[cv_score(LR_Pipeline),cv_score(SVC_Pipeline),cv_score(RF_Pipeline),cv_score(XGBC_Pipeline)]})
CV_scores.index = ['LR_Pipeline','SVC_Pipeline','RF_Pipeline','XGBC_Pipeline']
print(CV_scores)
for C in np.logspace(-4, 4, num=10):
    LR_Pipeline = make_pipeline(Imputer(), LogisticRegression(random_state=1,C=C))
    print('For C=%1f, the cross-validation score is %2f' %(C,cv_score(LR_Pipeline)))
for C in np.logspace(-4, 4, num=10):
    LR_Pipeline = make_pipeline(Imputer(), LogisticRegression(random_state=1,C=C,penalty='l1'))
    print('For C=%1f, the cross-validation score is %2f' %(C,cv_score(LR_Pipeline)))
reduced_test_data = test_data.drop(Columns_to_drop, axis=1)
reduced_test_data.head()
OHE_test_data = pd.get_dummies(reduced_test_data)
OHE_test_data.head(6)
XGBC_Pipeline.fit(X_training,y_training)
predictions = XGBC_Pipeline.predict(OHE_test_data)
print(predictions)
gender_submission = pd.read_csv("../input/gender_submission.csv")
print(gender_submission.head())

output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                       'Survived': predictions})
output.head(10)
output.to_csv('submission.csv', index=False)
