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

df = pd.read_csv("../input/" + filename, thousands=",")

print(df.dtypes)

df.head()
table = pd.pivot_table(df, values='Total', index=['Gender', 'Age_group','Year' ], columns=['Type'], aggfunc=np.sum)

table.head()
anyReason = np.sum(table, axis = 1)

anyReason
reasonsOrWays = table.columns.values

reasonsOrWays
youngWomen = table.loc['Female','15-29']
youngWomen
youngWomen.plot()
sortedTable = youngWomen.transpose().sort_values(by = 2012, ascending = False).head(10).transpose() 
top10Reasons = sortedTable.columns.values
youngWomen[top10Reasons].plot()
def plotForGroup(gender,age):

    tempTable = table.loc[gender,age]

    sortedTable = tempTable.transpose().sort_values(by = 2012, ascending = False).head(10).transpose() 

    top10Reasons = sortedTable.columns.values

    tempTable[top10Reasons].plot()

    plt.title("Top 10 suicide categories for " + gender + " "+ age + " in India")

    
plotForGroup('Female','15-29')
allAge = df['Age_group'].unique()

allGender = ['Male','Female']
for g in allGender:

    for a in allAge:

        plotForGroup(g,a)
plotForGroup('Male','15-29') 

plotForGroup('Male','30-44') 
plotForGroup('Male','45-59')
plotForGroup('Male','60+')