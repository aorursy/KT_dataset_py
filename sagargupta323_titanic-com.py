!pip install pycaret
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
import matplotlib.pyplot as plt
import seaborn as sns 
from pandas_profiling import ProfileReport
sns.set()
data=pd.read_csv('/kaggle/input/titanic/train.csv')
data.head()
data=data.drop(columns=['Name','Ticket'],index=1)
profile = ProfileReport(data, title="Pandas Profiling Report")
profile
data.Age.fillna(data.Age.mean(),inplace=True)
data.Embarked.fillna('S',inplace=True)
from pycaret.classification import *
data.isnull().sum()
from pycaret.classification import*
clf=setup(data,target='Survived',ignore_features=['PassengerId'],normalize=True,transformation = True,
          ignore_low_variance = True,numeric_features=['SibSp'] ,categorical_features=['Sex','Embarked','Pclass'],
          train_size=0.9)
compare_models()
catboost = create_model('catboost')
gbc = create_model('gbc')
tuned_catboost=tune_model('catboost')
tuned_gbc=tune_model('gbc')
plot_model(tuned_gbc,plot='confusion_matrix')
interpret_model(tuned_gbc)
interpret_model(tuned_catboost)
final_model=finalize_model(tuned_catboost)
test=pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()
predictions=predict_model(final_model,data=test)
predictions.head()
predictions.to_csv('predictions.csv')
