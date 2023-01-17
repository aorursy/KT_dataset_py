#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSipaK8Sl1f1KuRKMEKGB1b4vpPS_FSg0M9eWt1bLiPPS3xM2PP',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcS-dkw5Yt6VhMzP-C04S_esFI2F_BI94F06cjpJMRcZtP3_Z1FC',width=400,height=400)
df = pd.read_csv('../input/covid19-in-italy/covid19_italy_region.csv', encoding='ISO-8859-2')
df.head().style.background_gradient(cmap='PRGn')
df.dtypes
df["HospitalizedPatients"].plot.hist()

plt.show()
df["IntensiveCarePatients"].plot.box()

plt.show()
sns.pairplot(df, x_vars=['HomeConfinement'], y_vars='NewPositiveCases', markers="+", size=4)

plt.show()
df.corr()

plt.figure(figsize=(10,4))

sns.heatmap(df.corr(),annot=True,cmap='Blues')

plt.show()
fig, axes = plt.subplots(1, 1, figsize=(10, 4))

sns.boxplot(x='HospitalizedPatients', y='Deaths', data=df, showfliers=False);
fig=sns.lmplot(x="CurrentPositiveCases", y="Recovered",data=df)
import matplotlib.style



import matplotlib as mpl



mpl.style.use('classic')
fig=sns.lmplot(x="CurrentPositiveCases", y="Recovered",data=df)
for col in df.columns:

    plt.figure(figsize=(18,9))

    sns.factorplot(x=col,y='Recovered',data=df)

    plt.tight_layout()

    plt.show()
cat = []

num = []

for col in df.columns:

    if df[col].dtype=='O':

        cat.append(col)

    else:

        num.append(col)
num
plt.style.use('dark_background')

for col in df[num].drop(['TotalHospitalizedPatients'],axis=1):

    plt.figure(figsize=(12,7))

    plt.plot(df[col].value_counts(),color='Red')

    plt.xlabel(col)

    plt.ylabel('TotalHospitalizedPatients')

    plt.tight_layout()

    plt.show()
for col in df.columns:

    plt.figure(figsize=(18,9))

    sns.lineplot(x=col,y='Recovered',data=df)

    plt.tight_layout()

    plt.xlabel(col)

    plt.ylabel('')

    plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRwyOQlnntjHSJVyoImCWuUcWfEadWBkMLQhxixPvryz3BJQwi8',width=400,height=400)