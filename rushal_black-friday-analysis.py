# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
black_friday = pd.read_csv("../input/BlackFriday.csv")
black_friday.head()
black_friday.info()
black_friday.shape
if black_friday.isnull().any:
    print(black_friday.isnull().sum())
black_friday.Product_Category_2.fillna(value=0,inplace=True)
black_friday.Product_Category_3.fillna(value=0,inplace=True)

black_friday["Product_Category_2"] = black_friday.Product_Category_2.astype(int)
black_friday["Product_Category_3"] = black_friday.Product_Category_3.astype(int)
black_friday.isnull().sum()
#Sales on black friday depend on gender
plt.figure(figsize=(10,8))
sns.countplot(black_friday.Gender,palette='Set2')
#Sales on black friday depends on age groups
plt.figure(figsize=(14,9))
sns.countplot(black_friday.Age,palette='Set3',edgecolor='.6')
#Sales Based on age and gender
plt.figure(figsize=(14,10))
sns.countplot(x=black_friday.Age,hue=black_friday.Gender,palette='pastel',edgecolor='0.6')
#Sales based on Marital status
plt.figure(figsize=(8,7))
sns.countplot(x=black_friday.Age,hue=black_friday.Marital_Status,palette='Set3',edgecolor='.8')
black_friday['combined_Gender_marital'] =black_friday.apply(lambda x:'%s_%s' %(x.Gender,x.Marital_Status),axis=1)
black_friday.head()
plt.figure(figsize=(12,8))
sns.countplot(x=black_friday.Age,hue=black_friday.combined_Gender_marital,palette='Set3')
