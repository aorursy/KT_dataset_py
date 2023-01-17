# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#Loan Data into data frame
df_data = pd.read_csv("../input/loan.csv",low_memory=False)
#Group data by addr_state to start some analysis of counts by state
df_group = df_data.groupby('addr_state', as_index=False)['id'].count()
#Sort the grouping by counts to order ascending
df_group.sort_values(['id'], ascending=[True], inplace=True)
df_group = df_group.reset_index(drop=True)
#Plot data of counts by state shows CA way at top and ID at bottom x plot is in per 100000

plt.bar(df_group.index,df_group.id/100000)
plt.xticks(df_group.index,df_group.addr_state)
N = 3
params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )
plt.show()
#Plot data of totdal funded amount  by state shows CA way at top same as count and ID.TX and NYswap places interestingly in counts and amount at bottom x plot is in per 100000
df_group = df_data.groupby('addr_state', as_index=False).sum()
df_group.sort_values(['funded_amnt'], ascending=[True], inplace=True)
df_group = df_group.reset_index(drop=True)
plt.bar(df_group.index,df_group.funded_amnt/100000)
plt.xticks(df_group.index,df_group.addr_state)
N = 3
params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )
plt.show()
#Plot data of totdal funded amount by investors by state  shows no deviation from the funded amount by state
df_group.sort_values(['funded_amnt_inv'], ascending=[True], inplace=True)
df_group = df_group.reset_index(drop=True)
plt.bar(df_group.index,df_group.funded_amnt_inv/10000)
plt.xticks(df_group.index,df_group.addr_state)
N = 3
params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )
plt.show()
#A grouping by mean annual income of people who applied for loans in the state shows something interesting pattern of mean annual income of people statewise
df_group = df_data.groupby('addr_state', as_index=False).mean()
df_group.sort_values(['annual_inc'], ascending=[True], inplace=True)
df_group = df_group.reset_index(drop=True)
plt.bar(df_group.index,df_group.annual_inc/1000)
plt.xticks(df_group.index,df_group.addr_state)
N = 3
params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )
plt.show()
#A grouping by Issue date shows Lending clubs portfolio is growing and very rapidly in the last few months
df_group = df_data.groupby('issue_d', as_index=False).sum()
df_group.sort_values(['funded_amnt'], ascending=[True], inplace=True)
df_group = df_group.reset_index(drop=True)
plt.bar(df_group.index,df_group.funded_amnt/1000000)
plt.xticks(df_group.index,df_group.issue_d,rotation='vertical')
N = 4
params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )
plt.show()
#A plot of Grade and rate shows they are correlated
df_group = df_data.groupby('grade', as_index=False).mean()
df_group.sort_values(['int_rate'], ascending=[True], inplace=True)
df_group = df_group.reset_index(drop=True)
plt.bar(df_group.index,df_group.int_rate,align="center")
plt.xticks(df_group.index,df_group.grade)
N = 3
params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )
plt.show()
#A plot of sub grade and rate shows that they are also correlated
df_group = df_data.groupby('sub_grade', as_index=False).mean()
df_group.sort_values(['int_rate'], ascending=[True], inplace=True)
df_group = df_group.reset_index(drop=True)
plt.bar(df_group.index,df_group.int_rate,align="center")
plt.xticks(df_group.index,df_group.sub_grade)
N = 3
params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )
plt.show()
