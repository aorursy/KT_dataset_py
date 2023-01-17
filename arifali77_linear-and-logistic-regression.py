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
# From Scratch
import matplotlib.pylab as plt

from pylab import rcParams

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import scale

from collections import Counter

import seaborn as sns

%matplotlib inline
sns.set_style('whitegrid')

rcParams['figure.figsize']=8,6
enroll=pd.read_csv('../input/enroll-forecast/enrollment_forecast.csv')

enroll.head()
sns.pairplot(enroll)
print(enroll.corr())
enroll_data=enroll.iloc[:,2:4].values

enroll_target=enroll.iloc[:,1].values

enroll_data_names=['unem','hgrad']

x, y = scale(enroll_data), enroll_target
enroll.isnull().sum()
LinReg=LinearRegression(normalize=True)

LinReg.fit(x,y)

print(LinReg.score(x,y))
from sklearn.linear_model import LogisticRegression

from scipy.stats import spearmanr

from sklearn.model_selection import train_test_split

from sklearn import metrics
cars=pd.read_csv('../input/praactice-data/mt1cars.csv')

cars.head()
cars.rename(columns={'Unnamed: 0' : 'car_names'}, inplace=True)
cars_data = cars[['drat','carb']].values

y = cars[['am']]
sns.regplot(x='drat', y='carb', data=cars)
spearman_coefficient, p_value = spearmanr(cars.drat, cars.carb)

print('spearman_coefficient is %0.3f' % (spearman_coefficient))
cars.isnull().sum()
sns.countplot(x='am', data=cars, palette ='hls')
cars.info()
x=scale(cars_data)

LogReg=LogisticRegression()

LogReg.fit(x,y)

print(LogReg.score(x,y))
y_pred=LogReg.predict(x)

from sklearn.metrics import classification_report

print(classification_report(y,y_pred))