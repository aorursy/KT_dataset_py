!pip install pycaret
# Import dataset from pycaret repository

from pycaret.datasets import get_data

insurance = get_data('insurance')
# Initialize environment

from pycaret.regression import *

r1 = setup(insurance, target = 'charges', session_id = 123,

           normalize = True, silent = True,

           bin_numeric_features= ['age', 'bmi'])
# Compare Models

compare_models(fold=5)
# Train a linear regression model

model = create_model('xgboost')
tuneModel = tune_model('xgboost', optimize='mse', fold=5)
plot_model(tuneModel, plot = 'residuals')
plot_model(tuneModel, plot = 'error')
plot_model(tuneModel, plot = 'feature')
plot_model(tuneModel, plot = 'parameter')
pred_holdout = predict_model(tuneModel)
final_tuneModel = finalize_model(tuneModel)
final_tuneModel
# save transformation pipeline and model 

save_model(model, model_name = 'model_deploy_1')
save_experiment('experiment_1')
model
tuneModel
%ls -lrtah