!pip install --upgrade mxnet
!pip install --upgrade autogluon
!pip install -U scipy
import autogluon as ag
from autogluon import TabularPrediction as task
import pandas as pd
import numpy
import csv
data = pd.read_csv('/content/train.csv') 

predictor = task.fit(train_data=data,
                     label='Survived',
                     presets='best_quality',
                     eval_metric='accuracy', 
                     auto_stack=True)
test_data = pd.read_csv('/content/test.csv') 
prediction = predictor.predict(test_data)
sub = pd.read_csv('/content/gender_submission.csv')
sub['Survived'] = prediction
sub.to_csv('submission.csv', index=False)