# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install pycaret
train = pd.read_csv("/kaggle/input/train_fNxu4vz.csv")
test = pd.read_csv("/kaggle/input/test_fjtUOL8.csv")

train_id = train['Loan_ID']
train = train.drop('Loan_ID',axis=1)

test_id = test['Loan_ID']
test = test.drop('Loan_ID',axis=1)
print('train ids', train_id)
print('test ids', test_id)
train.head()
test.head()
from pycaret.classification import *
#intialize the setup
classifier =  setup(data = train, target = 'Interest_Rate',
                    normalize = True)



                 

compare_models(turbo=True)
#tuned_lightgbm = tune_model('lightgbm')

tuned_cat = tune_model('catboost')

#tuned_et = tune_model('et')
'''
et = create_model('et')
catboost = create_model('catboost')
ada = create_model('ada')
ridge = create_model('ridge')
lightgbm = create_model('lightgbm')

# stack trained models
stacked_models = stack_models(estimator_list=[et,catboost,ada,ridge,lightgbm])
'''

final_model = finalize_model(tuned_cat)
pred = predict_model(final_model, data=test)
submission = pd.DataFrame({'Loan_ID': test_id, 'Interest_Rate': pred['Label']})
submission.to_csv('submission_pycaret.csv', index = False)
