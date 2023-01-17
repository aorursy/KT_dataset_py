# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import matplotlib

import seaborn as sns
%matplotlib inline
from matplotlib.pyplot import rcParams
sns.set_style("darkgrid")
rcParams['figure.figsize']=20,9


# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/daily-inmates-in-custody.csv")
data.sample(5)
data.describe(include='all').T

data.info()
data.drop(['DISCHARGED_DT','SEALED'],axis=1,inplace=True)
plt.figure(figsize=(20,7))
sns.countplot(data['AGE'],palette='inferno')
plt.title("Distribution of Ages")
plt.xlabel("Age of Inmates")
plt.ylabel("Count")
plt.figure(figsize=(20,7))
sns.countplot(x='RACE', hue='GENDER', data=data, palette="inferno",
             order = data['RACE'].value_counts().index)
plt.ylabel("Number of Inmates")
plt.figure(figsize=(20,7))
sns.countplot(x='GENDER', hue='RACE', data=data, palette="inferno",
             order = data['GENDER'].value_counts().index)
plt.ylabel("Number of Inmates")
plt.figure(figsize=(20,7))
sns.countplot(x='SRG_FLG', hue='INFRACTION', data=data, palette="inferno",
             )
plt.ylabel("Number of Inmates")
