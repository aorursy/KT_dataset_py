# https://autogluon.mxnet.io/

!pip install --upgrade mxnet

!pip install autogluon==0.0.13b20200728
# https://github.com/awslabs/autogluon

from autogluon import TabularPrediction as task





train_data = task.Dataset(file_path='../input/titanic/train.csv')

test_data = task.Dataset(file_path='../input/titanic/test.csv')

predictor = task.fit(train_data=train_data, label='Survived',

                    num_bagging_folds=10,

                    stack_ensemble_levels=2,

                    presets='high_quality_fast_inference_only_refit',

                    time_limits=60*60*8)

y_pred = predictor.predict(test_data)

predictor.evaluate(train_data)

import pandas as pd





sub = pd.read_csv('../input/titanic/gender_submission.csv')

sub['Survived'] = y_pred
sub.to_csv('submission.csv', index=False)

sub.head()