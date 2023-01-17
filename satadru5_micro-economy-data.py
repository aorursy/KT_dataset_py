# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
eco=pd.read_csv("../input/macro.csv")
eco.head(3)
nan=pd.DataFrame(pd.isnull(eco).sum()).reset_index()

nan.head(3)
na=eco.isnull().sum().sort_values(ascending=False)
na
eco.shape
eco.head(3)
sns.violinplot(eco['gdp_annual'])
corr=eco.corr()

corr = (corr)

plt.figure(figsize=(14,14))

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws=



{'size': 15},

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.plt.title('Heatmap of Correlation Matrix')
corr.gdp_annual.sort_values(ascending=False)
eco=eco.fillna(0)
sns.boxplot(eco['salary'])
sns.pairplot(eco[['deposits_value','gdp_annual','rent_price_2room_bus']])
cor_gdp=eco[['deposits_value','gdp_annual','rent_price_2room_bus']].corr()
cor_gdp
sns.regplot(x='deposits_value',y='gdp_annual',data=eco)
sns.boxplot(eco['deposits_value'])