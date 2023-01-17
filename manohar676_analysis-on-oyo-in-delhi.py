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

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split
Oyo_Data = pd.read_csv("../input/oyorooms.csv",index_col="Unnamed: 0")

Oyo_Data.head()
Oyo_Data.shape
#####    Droping   DateOfRegistration ,DateOfResolution   columns

DATA = Oyo_Data.drop(['% DISCOUNT','REGION','HOTEL NAME','LOCALITY'], axis =1)

DATA_Numeric = DATA.replace(to_replace={'STATUS': {'Available': 1,'SOLD OUT':0}})

DATA_Numeric.head()
DATA_Numeric.dtypes
DATA_Numeric['OLD PRICE'] = DATA_Numeric['OLD PRICE'].str.replace('₹', ' ')

DATA_Numeric['PRICE AFTER DISCOUNT'] = DATA_Numeric['PRICE AFTER DISCOUNT'].str.replace('₹', ' ')



DATA_Numeric.head()
df = DATA_Numeric.dropna(how='any',axis=0) 

df.isnull().sum(axis = 0)

df.dtypes
df[['OLD PRICE','PRICE AFTER DISCOUNT']] = df[['OLD PRICE','PRICE AFTER DISCOUNT']].apply(pd.to_numeric, errors ='ignore')

df.dtypes
df["Price_Flucations"] = (df["OLD PRICE"]- df["PRICE AFTER DISCOUNT"])  

df.head()
### Bar_Plot of Categorical Coloumn in Sub-Reasons

sns.set(font_scale=1.0)

plt.figure(figsize=(18,18))

sns.countplot(data = Oyo_Data, y = 'OLD PRICE')
### Bar_Plot of Categorical Coloumn in Sub-Reasons

sns.set(font_scale=1.0)

plt.figure(figsize=(18,18))

sns.countplot(data = Oyo_Data, y = 'PRICE AFTER DISCOUNT')
Oyo_Data["STATUS"].value_counts(331).plot(kind='pie',figsize= (6,6));
sns.set(font_scale=1.0)

plt.figure(figsize=(15,15)) 

sns.countplot(data = df, y = 'Price_Flucations')