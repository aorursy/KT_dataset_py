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
df = pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/train.csv")

df.sample(10)
df.info()
df_test = pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/test.csv")

df_test.sample(10)
df['Response'].value_counts()/len(df)*100
df['Gender'].unique()
n_males = len(df[df['Gender']=="Male"])

n_females = len(df)-n_males

df.groupby('Gender').sum()['Response']/np.array([n_males,n_females])*100
sns.distplot(df[df['Response']==1]['Age'], kde=False)

plt.show()
df['Region_Code'].unique()
df['Region_Code'] = df['Region_Code'].apply(lambda x: str(int(x)))

df['Region_Code'].sample(2)
fig = plt.figure()

fig.set_size_inches(15,10)

region = df.groupby('Region_Code').sum().sort_values(by="Response",ascending = False).reset_index()

region = region[['Region_Code','Response']]

region.head(3)
fig = plt.figure()

fig.set_size_inches(15,5)

plt.bar(region.Region_Code, region.Response)

plt.title("Response amongst different regions")

plt.xlabel("Region Code")

plt.ylabel("Number of positive Response")

plt.show()