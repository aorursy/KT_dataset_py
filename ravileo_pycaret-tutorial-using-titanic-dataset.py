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
!pip3 install pycaret
import pandas as pd
from pycaret import classification
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
classification_setup = classification.setup(data=train,target='Survived')
classification.compare_models()
# Create Xgboost model
classification_xgb = classification.create_model('xgboost')
# Tune the model
tune_xgb = classification.tune_model('xgboost')
# build the lightgbm model
classification_lightgbm = classification.create_model('lightgbm')
# Tune lightgbm model
tune_lightgbm = classification.tune_model('lightgbm')
# Residual Plot
classification.plot_model(tune_lightgbm)
# Error Plot
classification.plot_model(tune_lightgbm, plot = 'error')
# Feature Important plot
classification.plot_model(tune_lightgbm, plot='feature')
# Evaluate model
classification.evaluate_model(tune_lightgbm)
# read the test data
test_data_classification = pd.read_csv("/kaggle/input/titanic/test.csv")
# make predictions
predictions = classification.predict_model(tune_xgb, data=test_data_classification)
# view the predictions
predictions
# read the test data
test_data_classification = pd.read_csv("/kaggle/input/titanic/test.csv")
# make predictions
predictions = classification.predict_model(tune_lightgbm, data=test_data_classification)
# view the predictions
predictions