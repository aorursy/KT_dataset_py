# run this cell of code to install pycaret

!pip install pycaret
# check version (after installation)

from pycaret.utils import version

version()
from pycaret.datasets import get_data

diabetes = get_data('diabetes')
from pycaret.classification import *

exp1 = setup(diabetes, target = 'Class variable', silent = True) #silent True for unattended run during kernel
compare_models()
adaboost = create_model('ada')
tuned_adaboost = tune_model('ada')
# creating a decision tree model

dt = create_model('dt')



# ensembling a trained dt model

dt_bagged = ensemble_model(dt)
# AUC plot

plot_model(adaboost, plot = 'auc')
evaluate_model(adaboost)
# create a model

xgboost = create_model('xgboost')



# summary plot

interpret_model(xgboost)
# correlation plot

interpret_model(xgboost, plot = 'correlation')
interpret_model(xgboost, plot = 'reason', observation = 0) 
# create a model

rf = create_model('rf')



# predict test / hold-out dataset

rf_holdout_pred = predict_model(rf)
predictions = predict_model(rf, data = diabetes)

predictions.head()
# AWS security must be configured on local pc before running the below code:



# deploy_model(model = rf, model_name = 'rf_aws', platform = 'aws', authentication =  {'bucket'  : 'pycaret-test'})
# saving model

save_model(adaboost, model_name = 'ada_for_deployment')
save_experiment(experiment_name = 'my_first_experiment')