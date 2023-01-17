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
df=pd.read_csv('/kaggle/input/titanic/train.csv')
df.head(3)
!pip install pycaret
from pycaret.classification import*
clf1=setup(data=df,target='Survived')
compare_models()
gbc=create_model('catboost')
tune_gbc=tune_model('catboost')
gcb_final=finalize_model(tune_gbc)
dt=pd.read_csv('/kaggle/input/titanic/test.csv')
ds=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
predict=predict_model(gcb_final,data=dt)
dt.head(3)
predict.head(3)
predict_final=predict['Label']
ds['Survived']=predict_final
ds
