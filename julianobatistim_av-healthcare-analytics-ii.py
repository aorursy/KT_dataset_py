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
raw = pd.read_csv('/kaggle/input/av-healthcare-analytics-ii/healthcare/train_data.csv')

raw = raw.dropna()

raw.head()
raw.shape
from sklearn.model_selection import train_test_split
train, test = train_test_split(raw, test_size=0.05)
train.shape
test.shape
!pip install pycaret
from pycaret.classification import *

clf1 = setup(
    train, 
    target = 'Stay',
    ignore_features = ['case_id', 'patientid', 'Visitors with Patient'],
    session_id=1945,
    # normalize = True, 
    # transform_target = True, 
    # polynomial_features = True, 
    # feature_selection = True, 
    # train_size=0.7,
    categorical_features=['City_Code_Patient', 'Hospital_code', 'Bed Grade'], 
    # log_experiment=True,
    # log_plots=True,
    use_gpu=True,
    # experiment_name='av-healthcare-analytics-ii-ex-v1'
    silent = True
)
# best = compare_models(fold = 5)

best = create_model('lightgbm')
plot_model(best)
plot_model(best, plot='confusion_matrix')
evaluate_model(best)
tunned = tune_model(best)
ensembled = ensemble_model(tunned)
plot_model(ensembled)
plot_model(ensembled, plot='confusion_matrix')
evaluate_model(ensembled)
predict_test = predict_model(ensembled, test)
predict_test = predict_test.dropna()
predict_test.to_csv('predict_test.csv', index=False)
predict_test.head()
predict_test['comp'] = np.where(predict_test['Stay'] == predict_test['Label'], 'Correct', 'Incorrect')
predict_test.groupby('comp').count()['Label']
print(predict_test.groupby('comp').count()['Label'][0] / predict_test.groupby('comp').count()['Label'][1])
submit = pd.read_csv('/kaggle/input/av-healthcare-analytics-ii/healthcare/test_data.csv')
predict_submit = predict_model(ensembled, submit)
predict_submit
predict_submit_format = pd.DataFrame({ 'case_id': predict_submit['case_id'], 'Stay': predict_submit['Label']})
predict_submit_format.to_csv('Submission.csv', index=False)
predict_submit_format
finalize_model(ensembled)
save_model(ensembled, 'model')