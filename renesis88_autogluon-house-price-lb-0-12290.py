!pip install --upgrade mxnet
!pip install --upgrade autogluon
!pip install -U ipykernel
import autogluon as ag
from autogluon import TabularPrediction as task
import pandas as pd
import numpy as np
import csv
dataset = pd.read_csv('/content/train.csv') 
# dataset = task.Dataset('data/train')
dataset.head()
dataset.dtypes
# Convert all NaN to Zeros
dataset.fillna(0, inplace=True)
metric="root_mean_squared_error"

# Evaluation Metric Options Below :
# 'accuracy’, ‘balanced_accuracy’, ‘f1’, ‘f1_macro’, ‘f1_micro’, ‘f1_weighted’, ‘roc_auc’, ‘average_precision’
# ‘precision’, ‘precision_macro’, ‘precision_micro’, ‘precision_weighted’, ‘recall’, ‘recall_macro’, ‘recall_micro’
# ‘recall_weighted’, ‘log_loss’, ‘pac_score’
# For purpose of scoring for Kaggle, using Root Mean Square Log Error (RMSLE), work around in AutoGluon as follows :

dataset['SalePrice'] = np.log(dataset['SalePrice'])
predictor = task.fit(train_data=dataset,
                     label='SalePrice',
                     presets='best_quality', 
                     eval_metric= metric, 
                     verbosity=2)
new_data = pd.read_csv('/content/test.csv') 
new_data.fillna(0, inplace=True)
prediction = predictor.predict(new_data)
# Inverse transform from Root Mean Squared Log Error (RMSLE)
prediction = np.exp(prediction)
sub = pd.read_csv('/content/sample_submission.csv')
sub['SalePrice'] = prediction
sub.to_csv('submission.csv', index=False)