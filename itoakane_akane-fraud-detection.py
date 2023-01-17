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
train = pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/train.csv')

test = pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/test.csv')
X = train.drop('Class', axis = 1).values

y = train['Class'].values
from imblearn.over_sampling import SMOTE



smote = SMOTE(kind='svm')

x_resampled, y_resampled = smote.fit_sample(X, y)
import xgboost as xgb

model = xgb.XGBClassifier()

model.fit(x_resampled , y_resampled)
X = test.values

predict = model.predict(X)

predict_proba = model.predict_proba(X)
submit = pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/sampleSubmission.csv')

submit['Class'] = predict

submit.to_csv('@akane-fraud-detection-in-credit-card-XGBoost.csv', index=False)