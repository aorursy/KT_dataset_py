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
train = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')
train.head()
test.head()
train.shape
test.shape
train.info()
test.info()
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
Fatalities = train["Fatalities"].groupby(train['Date']).sum()

df= pd.DataFrame(Fatalities)

df.plot(kind='bar', color="blue")
ConfirmedCases = train["ConfirmedCases"].groupby(train['Date']).sum()

df = pd.DataFrame(ConfirmedCases)

df.plot(kind='bar', color="red")

plt.show()
ConfirmedCases = train[train["Country_Region"]=='China']["ConfirmedCases"].groupby(train['Date']).sum()

df = pd.DataFrame(ConfirmedCases)

df.plot(kind='bar', color="red")

plt.show()
Fatalities = train[train["Country_Region"]=='China']["Fatalities"].groupby(train['Date']).sum()

df= pd.DataFrame(Fatalities)

df.plot(kind='bar', color="blue")