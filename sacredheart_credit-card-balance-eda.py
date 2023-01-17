# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
credit_card = pd.read_csv('../input/credit_card_balance.csv')
credit_card.head()
credit_card.shape
def m(credit_card):
    values = credit_card.columns[credit_card.isnull().any()].tolist() 
    return values

df=credit_card[m(credit_card)].isnull().sum()/credit_card.shape[0]
df=df.to_frame().reset_index()
df

credit_card.columns
credit_card.iloc[3075:3080,1:8]
credit_card.fillna(credit_card["AMT_DRAWINGS_ATM_CURRENT"].mean(), inplace=True)
credit_card.fillna(credit_card["AMT_DRAWINGS_OTHER_CURRENT"].mean(), inplace=True)
credit_card.fillna(credit_card["AMT_DRAWINGS_POS_CURRENT"].mean(), inplace=True)
credit_card.fillna(credit_card["AMT_INST_MIN_REGULARITY"].mean(), inplace=True)
credit_card.fillna(credit_card["AMT_PAYMENT_CURRENT"].mean(), inplace=True)
credit_card.fillna(credit_card["CNT_DRAWINGS_ATM_CURRENT"].mean(), inplace=True)
credit_card.fillna(credit_card["CNT_DRAWINGS_OTHER_CURRENT"].mean(), inplace=True)
credit_card.fillna(credit_card["CNT_DRAWINGS_POS_CURRENT"].mean(), inplace=True)
credit_card.fillna(credit_card["CNT_INSTALMENT_MATURE_CUM"].mean(), inplace=True)


credit_card.iloc[42:50,13:17]
# plt.axes((left, bottom, width, height), facecolor='w')
ax = sns.distplot(credit_card["CNT_INSTALMENT_MATURE_CUM"])

ax = sns.barplot(x=df.columns[0], y=df.columns[1], data= df)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
credit_card.boxplot(figsize=(20,8))
plt.show()
