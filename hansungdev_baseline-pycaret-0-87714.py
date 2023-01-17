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
# path = '../input/'
path = '/kaggle/input/kakr-4th-competition/'
import os
os.listdir(path)
#!pip install pycaret
import pandas as pd
from pycaret.classification import *

pd.set_option('max_columns', 500)
pd.set_option('max_rows', 500)
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')
df = train.copy()
print(train.shape)
print(test.shape)
print(submission.shape)
display(train.head())
display(test.head())
display(submission.head())
# test 0.3

from pycaret.classification import *
exp1 = setup(df, target = 'income',
            ignore_features=['id'])
# F1 score
# sort: string, default = ‘Accuracy’
# The scoring measure specified is used for sorting the average score grid. 
# Other options are ‘AUC’, ‘Recall’, ‘Precision’, ‘F1’, ‘Kappa’ and ‘MCC’.

best_3 = compare_models(sort = 'F1', n_select = 3)
lightgbm = create_model('lightgbm')
# models()
# models(type='ensemble').index.tolist()
tuned_lightgbm = tune_model(lightgbm)
plot_model(estimator = tuned_lightgbm, plot = 'auc')
plot_model(estimator = tuned_lightgbm, plot = 'confusion_matrix')
plot_model(estimator = tuned_lightgbm, plot = 'feature')
plot_model(estimator = tuned_lightgbm, plot = 'class_report')
# SHAP
interpret_model(lightgbm)
interpret_model(lightgbm, plot = 'correlation')

# threshold = 0.34
evaluate_model(tuned_lightgbm)
pred_holdouts = predict_model(lightgbm)
pred_holdouts.head()
display(train.shape)
display(pred_holdouts.shape)


lightgbm_final = finalize_model(lightgbm)
predictions = predict_model(lightgbm_final, test)
display(train.shape)
display(predictions.shape)
# tuned_lightgbm

# lightgbm_final = finalize_model(tuned_lightgbm)
# predictions = predict_model(lightgbm_final, test)
predictions.head()
submission['prediction'] = predictions['Score']
for ix, row in submission.iterrows():
    if row['prediction'] > 0.5:
        submission.loc[ix, 'prediction'] = 1
    else:
        submission.loc[ix, 'prediction'] = 0
submission = submission.astype({"prediction": int})
submission.to_csv('submission_T-AcademyXKaKr-lightgbm-Pycaret#03.csv', index=False)
submission.head()
