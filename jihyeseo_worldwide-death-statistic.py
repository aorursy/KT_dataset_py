# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import matplotlib 

import matplotlib.pyplot as plt

import sklearn

%matplotlib inline

import matplotlib.pyplot as plt 

plt.rcParams["figure.figsize"] = [16, 12]

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

filename = check_output(["ls", "../input"]).decode("utf8").strip()

df = pd.read_csv("../input/" + filename, thousands=",", skipfooter = 4)

print(df.dtypes)

df.head()
df.tail(5)
years = df['Year'].unique()

years
reasons = df['Cause of death (WHO)'].unique()

ages = df['Age'].unique() 

countries = df['Country or Area'].unique()

genders = df['Sex'].unique()
table = pd.pivot_table(df, values='Value', index=['Age','Sex','Country or Area'], columns=['Cause of death (WHO)'], aggfunc=np.sum)

table.head()
countries
dt = table.loc['Total']
tempTable = dt.loc['Female','Australia']

sortedTable = tempTable.transpose().sort_values(ascending = False).head(10).transpose()

sortedTable
tempTable = dt.loc['Male','South Africa']

sortedTable = tempTable.transpose().sort_values(ascending = False).head(10).transpose()

sortedTable