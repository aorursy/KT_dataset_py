!pip install pycaret
%pwd
INPUT_PATH='/kaggle/input/liberty-mutual-group-property-inspection-prediction'
%ls -lrtah $INPUT_PATH
import pandas as pd
df_Train = pd.read_csv(f'{INPUT_PATH}/train.csv.zip')

df_Test = pd.read_csv(f'{INPUT_PATH}/test.csv.zip')

df_Sample_Sub = pd.read_csv(f'{INPUT_PATH}/sample_submission.csv.zip')
df_Train.shape
df_Train.sample(10)
# Initialize environment

from pycaret.regression import *

r1 = setup(df_Train, target = 'Hazard', session_id = 25,

           normalize = True, silent = True,

           sampling = False)
# Compare Models

compare_models(blacklist=['svm', 'lar', 'llar', 'ransac', 'par', 'tr', 'knn', 'en', 'huber'], turbo=True, fold=5)
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