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
df = pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv")
df.head(2)
!pip install pycaret
from pycaret.regression import *
data = df.sample(frac=0.9, random_state=786).reset_index(drop=True)
data_unseen = df.drop(df.index).reset_index(drop=True)

print('Data for Modeling: ' + str(df.shape))
print('Unseen Data For Predictions: ' + str(df.shape))
exp_reg101 = setup(data = data, target = 'price', session_id=123) 
# compare_models()
catboost = create_model('catboost')
tuned_catboost = tune_model('catboost')
predict_model(tuned_catboost)
final_tuned_catboost = finalize_model(tuned_catboost)
predict_model(final_tuned_catboost);
data_n=data.drop("price",axis =1)
data_n.head()
data.head()
unseen_predictions = predict_model(final_tuned_catboost, data=data_n,round=0)
unseen_predictions.head()