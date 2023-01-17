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
#predict severity/ correlation between severity and location/weather/etc../PCA column reduction/
us_acc = pd.read_csv("../input/us-accidents/US_Accidents_Dec19.csv")

us_acc.head()
us_acc.info()
us_acc.groupby('Severity').count()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.countplot(x='Source', data=us_acc)

plt.title('frequences by sources')
sns.countplot(x='Severity', data=us_acc)

plt.title('frequences of Severity')
plt.figure(figsize=(20,8))

sns.countplot(x='State', data=us_acc, order=us_acc.State.value_counts().index)

plt.show()
#top 10 accidents state

sns.countplot(x='State', data=us_acc, order=us_acc.State.value_counts().iloc[:10].index)


ct = pd.crosstab(us_acc.State, us_acc.Severity)

ct.plot.bar(stacked=True)





us_acc.plot(kind = "scatter", x="Start_Lng",y="Start_Lat",alpha = 0.009)


us_acc.plot(kind = "scatter", x="Start_Lng",y="Start_Lat",alpha = 0.009,c="Severity", 

            cmap=plt.get_cmap("jet"), colorbar = False, figsize=(15,7))

plt.figure(figsize=(20,12))

plt.show()