# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np
data=pd.read_csv("../input/start-up/50_Startups.csv")
data.head()
data.isnull()
data['R&D Spend'].isnull().sum()
data.isnull().sum()
data.fillna(999).head()
data.fillna(method="bfill").head()
data.interpolate().head()
data.replace(np.nan,-999).head()
from sklearn.preprocessing import Imputer
imputer=Imputer(strategy='mean')
data_new=data.drop("State",axis=1)
data_new.head()
imputer.fit_transform(data_new)
import seaborn as sns

import numpy  as np

import matplotlib.pyplot  as plt
win_data=pd.read_csv("../input/wine-quqlity/winequality-red.csv")
win_data.head()
#visualize a outlier in fixed acidity
sns.boxplot(win_data["fixed acidity"])
win_data.head()
win_data['fixed acidity'].unique()
from scipy import stats
z_score=np.abs(stats.zscore(win_data))
z_score
print(np.where(z_score>3))
print(z_score[13][9])
#shape of our original data

win_data.shape
clean_data=win_data[(z_score<3).all(axis=1)]
clean_data.shape
clean_data['fixed acidity'].unique()
clean_data.head(5)
q1=win_data.quantile(0.25)

q2=win_data.quantile(0.75)

print(q1)

print(q2)
IQR=q2-q1

clean_data2=win_data[((win_data<(q1-1.5*IQR)) | (win_data > (q2+1.5*IQR))).any(axis=1)]
clean_data2.shape
clean_data2=win_data[~((win_data<(q1-1.5*IQR)) | (win_data > (q2+1.5*IQR))).any(axis=1)]
clean_data2
clean_data2.shape
sns.boxplot(clean_data2['fixed acidity'])