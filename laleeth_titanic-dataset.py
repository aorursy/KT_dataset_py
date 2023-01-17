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
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.head()
!pip install pycaret
from pycaret.classification import *
clf1 = setup(df,target='Survived',ignore_features=['Name','Ticket','PassengerId'])
compare_models()
tuned_catboost = tune_model('lightgbm',optimize='AUC')
evaluate_model(tuned_catboost)
final_lightgbm=finalize_model(tuned_catboost)
print(final_lightgbm)
test =pd.read_csv('/kaggle/input/titanic/test.csv')
predictions = predict_model(final_lightgbm,data=test)
predictions.head()