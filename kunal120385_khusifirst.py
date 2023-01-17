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
trainDS = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

testDS  = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')
#Train Head 

trainDS.head()
#Test Head

testDS.head()
#Change the name in Test and Train data Set

testDS.rename(columns = {'Province_State':'Province','Country_Region':'Country'},inplace=True)

testDS.head()
trainDS.rename(columns = {'Province_State':'Province','Country_Region':'Country'},inplace=True)

trainDS.head()
testDS.info()
trainDS.info()
testDS['Country'].unique()
testDS['Country'].value_counts()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

trainDS.columns

testDS.columns
confirmed  = testDS.where(testDS.ConfirmedCases!=0).dropna(axis=0)

confirmed.groupby('Country').ConfirmedCases.sum().plot(kind='barh', figsize=(10,5))

plt.show()
fatalities  = testDS.where(testDS.Fatalities!=0).dropna(axis=0)

fatalities.groupby('Country').Fatalities.sum().plot(kind='barh', figsize=(10,5))

plt.show()