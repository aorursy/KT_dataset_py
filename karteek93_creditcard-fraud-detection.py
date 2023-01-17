# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!pip install pycaret
data=pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
data.head()
data.isna().sum()
data.isnull().sum()
data.nunique()
data.describe().transpose()
len(data[data["Class"]==0])
len(data[data["Class"]==1])
from pycaret.classification import *
classify=setup(data=data,target="Class")
compare_models()
# getting the catboost model
catboost=create_model('catboost')