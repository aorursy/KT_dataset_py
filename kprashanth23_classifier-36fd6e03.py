# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train_data=pd.read_csv("../input/chronic-kidney-disease/kidney_disease_train.csv")
df_test_data=pd.read_csv("../input/chronic-kidney-disease/kidney_disease_test.csv")
df_train_data.head()
df_train_data.classification.unique()
df_train_data.info()
df_train_data.drop("id",axis=1,inplace=True) 
df_test_data.drop("id",axis=1,inplace=True) 
df_train_data.head()
df_train_data.classification=[1 if each=="ckd" else 0 for each in df_train_data.classification]
df_train_data.isnull().sum() 
df_test_data.isnull().sum() 
df_train_data.wc.unique()
df_test_data['classification']=99
df_data = pd.concat([df_train_data, df_test_data])
df_data.head()
df_data.classification.unique()
