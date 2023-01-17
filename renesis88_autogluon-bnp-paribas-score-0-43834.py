!pip install mxnet-cu101
!pip install autogluon
import pandas as pd
import numpy as np
import csv
import autogluon as ag
from autogluon import TabularPrediction as task
dataset = pd.read_csv('../input/bnp-paribas-cardif-claims-management/train.csv.zip')
test_dataset = pd.read_csv('../input/bnp-paribas-cardif-claims-management/test.csv.zip')
dataset.head()
metric="log_loss"

# Evaluation Metric for Classification Below :
# 'accuracy’, ‘balanced_accuracy’, ‘f1’, ‘f1_macro’, ‘f1_micro’, ‘f1_weighted’, ‘roc_auc’, ‘average_precision’
# ‘precision’, ‘precision_macro’, ‘precision_micro’, ‘precision_weighted’, ‘recall’, ‘recall_macro’, ‘recall_micro’
# ‘recall_weighted’, ‘log_loss’, ‘pac_score’

# Metric for Regression Below :
#‘root_mean_squared_error’, ‘mean_squared_error’, ‘mean_absolute_error’, ‘median_absolute_error’, ‘r2’]

quality="best_quality"

# ‘best_quality’, ‘best_quality_with_high_quality_refit’, ‘high_quality_fast_inference_only_refit’
# ‘good_quality_faster_inference_only_refit’, ‘medium_quality_faster_train’, ‘optimize_for_deployment’, ‘ignore_text’

# visual='mxboard'

# ‘mxboard’, ‘tensorboard’, ‘none’

# dir = '/home/andrew/Documents/PropertyPrices' # specifies folder where to store trained models

problem ='binary'
#['binary', 'multiclass', 'regression'])
predictor = task.fit(train_data=dataset,
                     label='target',
                     presets=quality,
                     problem_type=problem,
                     eval_metric= metric,
                     auto_stack=True,
                     ngpus_per_trial=1,
                     verbosity=2)
results = predictor.fit_summary()
prediction = predictor.predict_proba(test_dataset)
sub = pd.read_csv('../input/bnp-paribas-cardif-claims-management/sample_submission.csv.zip')
sub['PredictedProb'] = prediction
sub.to_csv('./submission.csv', index=False)