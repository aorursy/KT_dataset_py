# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!pip install pycaret
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.dropna(inplace=True)
train.info()
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.shape
train['Cabin'].unique()
from pycaret.classification import *
exp_clf101 = setup(data = train, target = 'Survived', session_id=123) 
compare_models()
lr = create_model('lr')
tune_lr = tune_model('lr')
print(tune_lr)
predict_model(tune_lr);
final_lr = finalize_model(tune_lr)
predict_model(final_lr);
unseen_predictions = predict_model(final_lr, data=test)
unseen_predictions.head(20)
unseen_predictions.to_csv('test_predict.csv')
unseen_predictions.shape
unseen_predictions.head(5)
