import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

# import data from CSV file

df = pd.read_csv("../input/Canada_PermanentResidents.csv")
df.head()
# update 3 row index as country

df.iloc[3,0] = 'Country'
# take column names from row 3

df.columns = df.iloc[3,:]
# remove first 3 rows 

df = df.drop([0,1,2,3],axis =0)
# set country name as index

df = df.set_index("Country")
# tail records are having invalid data

df.tail(8)
# removing tail records including totals for each column

df = df.iloc[:-8,]
df.head()
# created method for update column names to fetch monthly data, quarterly data and yearly data

def coll(s):

    dt = []

    year = '2015'

    for a in s:

        if str(a) == 'nan':

            dt.append(year)

            year = str(int(year)+1)

        else:

            if "Total" in a:

                dt.append(a[:2]+'_'+year)

            else:

                dt.append(a+'_'+year)

    return dt
# update column names

df.columns = coll(df.columns)
df.head()
# few data elements are '--' for 0 values and have ',' for thousends, change data valuce to numaric

def fill(col):

    for cl in col:

        df[cl] = df[cl].apply(lambda x: '0' if x == '--' else x)

        df[cl] = df[cl].apply(lambda x: str(x).replace(',', ''))

        df[cl] = pd.to_numeric(df[cl], errors='ignore')
fill(df.columns)
df.head()
# data frame for quarterly data

df_Q = df[['Q1_2015','Q2_2015','Q3_2015','Q4_2015','Q1_2016','Q2_2016','Q3_2016','Q4_2016',

           'Q1_2017','Q2_2017','Q3_2017','Q4_2017','Q1_2018','Q2_2018','Q3_2018','Q4_2018','Q1_2019']]
# data frame for Year data

df_Y = df[['2015','2016','2017','2018','2019']]
# data frame for monthly data

df_M = df.drop(['Q1_2015','Q2_2015','Q3_2015','Q4_2015','Q1_2016','Q2_2016','Q3_2016','Q4_2016',

           'Q1_2017','Q2_2017','Q3_2017','Q4_2017','Q1_2018','Q2_2018','Q3_2018','Q4_2018','Q1_2019',

                   '2015','2016','2017','2018','2019'], axis =1)
df_Q.head()
df_Y.head()
df_M.head()
# Sorting data by first calumn descending order

df_M.sort_values(by = df_M.columns[0],ascending = False,inplace=True)
# taking only top 5 counntry data to plot, change data from rows to columns to plot line plot

data_M = df_M.head().T

data_M.head()
cl = ['b','g','r','c','m','y','k','orange','gray','pink']

mk = ['.','o','v','^','<','>','1','2','3',','] 

plt.figure(figsize=(25, 20))

plt.grid()

n = 0

for col in data_M.columns:

    plt.plot(col,data=data_M, color=cl[n], marker=mk[n], linestyle='dashed',linewidth=2, markersize=12,label= col)

    n=n+1

plt.xlabel("Month")

plt.ylabel("No. of applications")

plt.title("Permanent Residents landed in Canada per month")

plt.xticks( rotation = 45, ha="right")

plt.legend() 
# Sorting data by first calumn descending order

df_Q.sort_values(by = df_Q.columns[0],ascending = False,inplace=True)
df_Q.head()
# taking only top 5 counntry data to plot, change data from rows to columns to plot line plot

data_Q = df_Q.head().T

data_Q.head()
cl = ['b','g','r','c','m','y','k','orange','gray','pink']

mk = ['.','o','v','^','<','>','1','2','3',','] 

plt.figure(figsize=(25, 20))

plt.grid()

n = 0

for col in data_Q.columns:

    plt.plot(col,data=data_Q, color=cl[n], marker=mk[n], linestyle='dashed',linewidth=2, markersize=12,label= col)

    n=n+1

plt.xlabel("Quarter")

plt.ylabel("No. of applications")

plt.title("Permanent Residents landed in Canada per quarter")

plt.xticks( rotation = 45, ha="right")

plt.legend() 
# Sorting data by first calumn descending order

df_Y.sort_values(by = df_Y.columns[0],ascending = False,inplace=True)
# taking only top 5 counntry data to plot, change data from rows to column to plot line plot

data_Y = df_Y.head().T

data_Y.head()
cl = ['b','g','r','c','m','y','k','orange','gray','pink']

mk = ['.','o','v','^','<','>','1','2','3',','] 

plt.figure(figsize=(15, 10))

plt.grid()

n = 0

for col in data_Y.columns:

    plt.plot(col,data=data_Y, color=cl[n], marker=mk[n], linestyle='dashed',linewidth=2, markersize=12,label= col)

    n=n+1

plt.xlabel("Year")

plt.ylabel("No. of applications")

plt.title("Permanent Residents landed in Canada per Year")

plt.xticks( rotation = 45, ha="right")

plt.legend() 