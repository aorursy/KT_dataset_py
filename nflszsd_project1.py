# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

% matplotlib inline

import seaborn as sns

import scipy.stats as st

sns.set(style="whitegrid", color_codes=True)

sns.set(font_scale=1)











#shape

train=pd.read_csv("../input/train.csv")

test =pd.read_csv("../input/test.csv")

print(train.shape)

print(test.shape)
#Missing Values

miss = train.isnull().sum()

miss = miss[miss > 0]

miss.sort_values(inplace=True)

miss.plot.bar()
#Basic Charactics 

train.describe()


y = train['SalePrice']

plt.figure(1); plt.title('Johnson SU')

sns.distplot(y, kde=False, fit=st.johnsonsu)

plt.figure(2); plt.title('Normal')

sns.distplot(y, kde=False, fit=st.norm)

plt.figure(3); plt.title('Log Normal')

sns.distplot(y, kde=False, fit=st.lognorm)
corrmat = train.corr()

top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]

g = sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")


plt.scatter(train["1stFlrSF"],train.SalePrice, color='blue')

plt.title("Sale Price wrt 1st floor")

plt.ylabel('Sale Price (in dollars)')

plt.xlabel("1st Floor in square feet");
sns.lmplot(x="1stFlrSF", y="SalePrice", data=train);