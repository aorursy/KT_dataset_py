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
raw = pd.read_csv('/kaggle/input/epitope-prediction/input_bcell.csv')

raw.head()
raw.columns
raw.isnull().sum()
for col in raw.columns:
    x = raw[col].unique()
    if len(x) < 20:
        print(f"{col}: {x}")
!pip install pycaret
from sklearn.model_selection import train_test_split
train, test = train_test_split(raw, test_size=0.05)
from pycaret.classification import *

experiment = setup(
    data = train 
    ,target = 'target'
    ,ignore_features = ['parent_protein_id', 'protein_seq', 'peptide_seq']
    ,normalize = True
)
best = compare_models()
save_model(best, 'model')
plot_model(best)
plot_model(best, 'confusion_matrix')
plot_model(best, 'threshold')
plot_model(best, 'pr')
plot_model(best, 'error')
plot_model(best, 'class_report')
plot_model(best, 'learning')
plot_model(best, 'manifold')
plot_model(best, 'calibration')
plot_model(best, 'vc')
plot_model(best, 'feature')
test_prediction = predict_model(best, test)
test_prediction = test_prediction.dropna()
test_prediction.to_csv('test_prediction.csv', index=False)
test_prediction
test_prediction['Label'] = test_prediction['Label'].apply(pd.to_numeric)
test_prediction['comp'] = np.where(test_prediction['target'] == test_prediction['Label'], 'Correct', 'Incorrect')
test_prediction.groupby('comp').count()['Label']
from sklearn.metrics import confusion_matrix

y_actu = test_prediction['target']
y_pred = test_prediction['Label']

cm = confusion_matrix(y_actu, y_pred)
cm
import seaborn as sn
sn.heatmap(cm, cmap="Blues", annot=True,annot_kws={"size": 16})
from sklearn.metrics import accuracy_score

print('VALIDATION ACCURACY', accuracy_score(y_actu, y_pred))
sars = pd.read_csv('/kaggle/input/epitope-prediction/input_sars.csv')

sars.head()
sars.columns
sars.isnull().sum()
sars_prediction = predict_model(best, sars)
sars_prediction = sars_prediction.dropna()
sars_prediction.to_csv('sars_prediction.csv', index=False)
sars_prediction
sars_prediction['Label'] = sars_prediction['Label'].apply(pd.to_numeric)
sars_prediction['comp'] = np.where(sars_prediction['target'] == sars_prediction['Label'], 'Correct', 'Incorrect')
sars_prediction.groupby('comp').count()['Label']
y_sars_actu = sars_prediction['target']
y_sars_pred = sars_prediction['Label']

cm_sars = confusion_matrix(y_sars_actu, y_sars_pred)
cm_sars
import seaborn as sn
sn.heatmap(cm_sars, cmap="Blues", annot=True)
print('VALIDATION SARS ACCURACY', accuracy_score(y_sars_actu, y_sars_pred))