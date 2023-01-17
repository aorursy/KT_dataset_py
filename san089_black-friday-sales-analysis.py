# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/BlackFriday.csv')
df.info()
df.head()
df.shape
gender_count = df['Gender'].value_counts()
gender_count.index = ["Male","Female"]
gender_count
gender_count.plot(kind = 'bar', title="Males and females on black friday sale.", legend="True")
user_purchase = df.loc[:, ['User_ID','Purchase']].groupby('User_ID').sum()
user_purchase.sort_values('Purchase', ascending = False).head(5)
sns.distplot(df.loc[:,'Purchase'])
stay_purchase_df = df.loc[: , ['Stay_In_Current_City_Years','Purchase']]
stay_purchase_data = stay_purchase_df.groupby('Stay_In_Current_City_Years').sum()
stay_purchase_data.plot(kind = 'bar', title="Purchases based on Stay in Current City in years")
df['Product_ID'].value_counts().head(5)
occupation_and_purchase = df.loc[:, ['Occupation','Purchase']]
oc = occupation_and_purchase.groupby('Occupation').sum()
sns.scatterplot(y = oc.Purchase, x = oc.index, data = oc)
data=df.loc[:,['Age','Gender', 'Purchase']].groupby(['Age','Gender']).sum()
data.reset_index(drop=False, inplace = True)
sns.barplot(x= data['Age'], y=data['Purchase'], data = data , hue = data['Gender'])