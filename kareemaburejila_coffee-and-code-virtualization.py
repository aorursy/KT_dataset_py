# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
file_path='../input/coffee-and-code/CoffeeAndCodeLT2018.csv'

coffee_df=pd.read_csv(file_path)

coffee_df.head()
coffee_df.columns
coffee_df.select_dtypes(include='object').columns
coffee_df.describe()
coffee_df.Gender.value_counts()
sns.pairplot(coffee_df)
sns.regplot(x=coffee_df['CodingHours'], y=coffee_df['CoffeeCupsPerDay'])
sns.pairplot(coffee_df, hue='CoffeeTime')

plt.figure(figsize=(16,10))

plt.title("Cofee Type")

sns.countplot(x="CoffeeType",data=coffee_df)
plt.figure(figsize=(16,8))

plt.title("Age Range")

sns.countplot(x="AgeRange",data=coffee_df,hue="Gender")