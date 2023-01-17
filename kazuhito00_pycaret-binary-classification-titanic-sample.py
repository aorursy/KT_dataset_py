!pip install pycaret 
import numpy as np 
import pandas as pd
import pandas_profiling
train_data = pd.read_csv('../input/titanic/train.csv')
test_data  = pd.read_csv('../input/titanic/test.csv')
submission_data = pd.read_csv('../input/titanic/gender_submission.csv')
train_data.info()
train_data.profile_report()
# Classification
from pycaret.classification import *
exp = setup(data=train_data, target='Survived', ignore_features = ['PassengerId', 'Name'], session_id=42) 
compare_models()
tuned_catboost_model = tune_model('catboost', optimize='AUC')
interpret_model(tuned_catboost_model)
blend_model = blend_models(fold=25)
ridge_model = create_model('ridge', verbose=False)
catboost_model = create_model('catboost', verbose=False)
lr_model = create_model('lr', verbose=False)
lgbm_model = create_model('lightgbm', verbose=False)
et_model = create_model('et', verbose=False)
rf_model = create_model('rf', verbose=False)
gbc_model = create_model('gbc', verbose=False)
xgboost_model = create_model('xgboost', verbose=False)
dt_model = create_model('dt', verbose=False)
ada_model = create_model('ada', verbose=False)
stack_model = stack_models([ridge_model, catboost_model, lr_model, lgbm_model, et_model, 
                            rf_model, gbc_model, xgboost_model, dt_model, ada_model], fold=20)
# Run the inference with the best performing model you've tried
predictions = predict_model(blend_model, data=test_data)
predictions.head()
submission_data['Survived'] = round(predictions['Label']).astype(int)
submission_data.to_csv('submission.csv',index=False)
submission_data.head()