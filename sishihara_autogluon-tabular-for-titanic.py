# https://autogluon.mxnet.io/

!pip install --upgrade mxnet

!pip install autogluon
# https://github.com/awslabs/autogluon

from autogluon import TabularPrediction as task





train_data = task.Dataset(file_path='../input/titanic/train.csv')

test_data = task.Dataset(file_path='../input/titanic/test.csv')

predictor = task.fit(train_data=train_data, label='Survived')

y_pred = predictor.predict(test_data)
import pandas as pd





sub = pd.read_csv('../input/titanic/gender_submission.csv')

sub['Survived'] = y_pred
sub.to_csv('submission.csv', index=False)

sub.head()