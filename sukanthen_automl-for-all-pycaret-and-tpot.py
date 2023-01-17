! pip install pycaret # Quite large depencies to install !
import numpy as np

import pandas as pd

from tpot import TPOTClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data = pd.read_csv('../input/titanic/train.csv')

data.head()
import pycaret

from pycaret.classification import *
clf1 = setup(data = data, 

             target = 'Survived',

             numeric_imputation = 'mean',

             categorical_features = ['Sex','Embarked'], 

             ignore_features = ['PassengerId','Name','Ticket','Cabin'],

             silent = True)
compare_models()
lgbm  = create_model('lightgbm')     
tuned_lgbm = tune_model('lightgbm')
plot_model(estimator = tuned_lgbm, plot = 'learning')
plot_model(estimator = tuned_lgbm, plot = 'feature')
plot_model(estimator = tuned_lgbm, plot = 'confusion_matrix')
# AUC Curve for Classifications models

plot_model(estimator = tuned_lgbm, plot = 'auc')
# Understand which feature had most role to play in the classification task

interpret_model(tuned_lgbm)
save_model(tuned_lgbm, 'Titaniclgbm')

# code to load the model for future uses or when making predictions

# trained_model = load_model('Titaniclgbm')
# Load the test data

test = pd.read_csv('../input/titanic/test.csv') 

predict_model(tuned_lgbm, data=test)
predictions = predict_model(tuned_lgbm, data=test)

predictions.head()
sub   = pd.read_csv('../input/titanic/gender_submission.csv')
sub['Survived'] = round(predictions['Score']).astype(int)

sub.to_csv('submission.csv',index=False)
# Blend your model ton other algorithm.

xgb   = create_model('xgboost');    

logr  = create_model('lr');   

blend = blend_models(estimator_list=[tuned_lgbm,logr,xgb])
data = pd.read_csv('../input/titanic/train.csv')

data.head()
data.isna().sum()
data =  data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

data['Sex'] = le.fit_transform(data.Sex)
replacer = {'S':2,'C':1,'Q':0}

data['Embarked'] = data['Embarked'].map(replacer)

data.head()
train = data.drop('Survived',axis=1)

test = data['Survived']

train.shape, test.shape
X_train, X_test, y_train, y_test = train_test_split(train,test,test_size=0.25)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
# Import tpot classifier and fit train and test datasets

# Set up Hyper-parameters of the TPOT Classifer such max_time_mins, which states maximum time for training the model through iterations of generations.

tpot = TPOTClassifier(verbosity=2, max_time_mins=10)



tpot.fit(X_train, y_train)

print(tpot.score(X_test, y_test))
# Check the specifications of the best fit algorith found out by TPOT

tpot.fitted_pipeline_
print(tpot.score(X_test, y_test))
tpot.export('TPOTSOLN.py')

# Check the output dir of the Notebook to your top-right. Download it and as a surprise you will see python code of your model with perfectly tuned hyper-parameters. 