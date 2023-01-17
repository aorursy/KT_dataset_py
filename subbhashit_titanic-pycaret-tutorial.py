!pip install pycaret
import numpy as np 
import pandas as pd 
train = pd.read_csv('../input/titanic/train.csv')
test  = pd.read_csv('../input/titanic/test.csv')
sub   = pd.read_csv('../input/titanic/gender_submission.csv')
from pycaret.classification import *
train.head()
train.info()
clf1 = setup(data = train, 
             target = 'Survived',
             numeric_imputation = 'mean',
             categorical_features = ['Sex','Embarked'], 
             ignore_features = ['Name','Ticket','Cabin'],
             silent = True)
compare_models()
lgbm  = create_model('lightgbm')  
tuned_lightgbm = tune_model(lgbm)
plot_model(estimator = tuned_lightgbm, plot = 'learning')
plot_model(estimator = tuned_lightgbm, plot = 'auc')
plot_model(estimator = tuned_lightgbm, plot = 'confusion_matrix')
plot_model(estimator = tuned_lightgbm, plot = 'feature')
evaluate_model(tuned_lightgbm)
interpret_model(tuned_lightgbm)
predict_model(tuned_lightgbm, data=test)
predictions = predict_model(tuned_lightgbm, data=test)
predictions.head()
sub['Survived'] = round(predictions['Score']).astype(int)
sub.to_csv('submission.csv',index=False)
sub.head()