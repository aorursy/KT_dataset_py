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
ep = pd.read_csv('../input/Ecommerce Purchases.csv')
ep.head()
ep.info()
ep['Purchase Price'].mean()
ep['Purchase Price'].max()
ep['Purchase Price'].min()
ep[ep['Language']=='en'].count()
ep[ep['Job']=='Lawyer'].info()
ep['AM or PM'].value_counts()
ep['Job'].value_counts().head()
ep[ep['Lot'] == "90 WT"]['Purchase Price']
ep[ep['Credit Card'] == 4926535242672853]['Email']
ep[(ep['CC Provider']=='American Express') & (ep['Purchase Price']>95)].count()
sum(ep['CC Exp Date'].apply(lambda x:x[3:]) == '25')
ep['Email'].apply(lambda x:x.split('@')[1]).value_counts().head()