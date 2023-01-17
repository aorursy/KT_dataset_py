# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")
train=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")
submission=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/submission.csv")
print(train.head(10))
train.isna().sum()
train.Province_State.unique()
labelencodermodel=LabelEncoder()
labe=LabelEncoder()
train['Country_Region']=labelencodermodel.fit_transform(train['Country_Region'])
train['Province_State']=labe.fit_transform(train['Province_State'])

