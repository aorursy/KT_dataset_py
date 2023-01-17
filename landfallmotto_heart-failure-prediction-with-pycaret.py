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
import missingno as msno

import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline
!pip install pycaret
from pycaret.classification import *
dataset=pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

dataset.head(5)
dataset.isnull().sum()
sns.countplot(dataset.DEATH_EVENT,palette='rainbow', alpha=0.75)
data = dataset.sample(frac=0.95, random_state=786)

data_unseen = dataset.drop(data.index)

data.reset_index(inplace=True, drop=True)

data_unseen.reset_index(inplace=True, drop=True)

print('Data for Modeling: ' + str(data.shape))

print('Unseen Data For Predictions: ' + str(data_unseen.shape))
clf1 = setup(data = data, target = 'DEATH_EVENT', session_id=55, silent=True,fix_imbalance = True)

best_model = compare_models(include=['dt','knn','mlp','rf','et','xgboost'])
print(best_model)
models()
model = create_model('xgboost')
#FIX: https://github.com/pycaret/pycaret/issues/377

mybooster = model.get_booster()

model_bytearray = mybooster.save_raw()[4:]

def myfun(self=None):return model_bytearray



mybooster.save_raw = myfun



interpret_model(model)
interpret_model(model, plot = 'reason', observation = 10)
tuned_model= tune_model(model)
plot_model(tuned_model, plot = 'auc')
plot_model(tuned_model, plot = 'pr')
plot_model(tuned_model, plot='feature')
plot_model(tuned_model, plot = 'confusion_matrix')
evaluate_model(tuned_model)
predict_model(tuned_model)