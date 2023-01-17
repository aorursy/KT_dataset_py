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
from pycaret.classification import *
df_train = '../input/health-insurance-cross-sell-prediction/train.csv'
df_test = '../input/health-insurance-cross-sell-prediction/test.csv'
import pandas as pd
df_train= pd.read_csv(df_train)
df_test= pd.read_csv(df_test)
df_train.shape
df_all = pd.concat([df_train, df_test], ignore_index=True)
data = df_all[:381109]
data_unseen = df_all[381109:]

data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)

print('train data for Modeling: ' + str(data.shape))
print('test data For Predictions: ' + str(data_unseen.shape))
df_all['Response'][:381109] = df_all['Response'][:381109].astype(str)
data = df_all[:381109]
data_unseen = df_all[381109:]

data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)

print('train data for Modeling: ' + str(data.shape))
print('test data For Predictions: ' + str(data_unseen.shape))
df_all.info()
clf = setup(data = data, target = 'Response', session_id=123)
best_model = compare_models()
Naive = create_model('nb') 
plot_model(Naive, plot = 'auc')
plot_model(Naive, plot = 'pr')
plot_model(Naive, plot = 'confusion_matrix')
test_data_predictions = predict_model(Naive, data=data_unseen)
test_data_predictions.head(10)
submission_file = pd.read_csv('../input/health-insurance-cross-sell-prediction/sample_submission.csv')
submission_file['Response'] = test_data_predictions['Label'] 
submission_file.to_csv('./my_submission1.csv',index=False)
et = create_model('et')
rf = create_model('rf')

blend_3_models = blend_models(estimator_list = [et,rf,Naive], method = 'soft')
