# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df=pd.read_excel('../input/Start-up Funding.xlsx')
df.head()
df.info()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.figure(figsize=(12,4))
sns.countplot('IndustryVertical',data=df, palette='rainbow')
market=df[df.IndustryVertical=='Marketplace']
plt.figure(figsize=(20,5))
sns.countplot('CityLocation',data=market,palette='RdBu_r')
plt.xticks(rotation=90)
sns.countplot('InvestmentType',data=df,palette='coolwarm')
plt.xticks(rotation=90)
food=df[df.IndustryVertical!='Food & Beverage']
plt.figure(figsize=(20,5))
sns.countplot('CityLocation',data=food, palette='rainbow')
plt.xticks(rotation=90)
tech=df[df.IndustryVertical!='Technology']
plt.figure(figsize=(20,5))
sns.countplot('CityLocation',data=tech, palette='rainbow')
plt.xticks(rotation=90)
health=df[df.IndustryVertical!='Healthcare']
plt.figure(figsize=(20,5))
sns.countplot('CityLocation',data=health, palette='rainbow')
plt.xticks(rotation=90)
plt.figure(figsize=(20,5))
sns.countplot('CityLocation',data=df,palette='RdBu_r')
plt.xticks(rotation=90)

