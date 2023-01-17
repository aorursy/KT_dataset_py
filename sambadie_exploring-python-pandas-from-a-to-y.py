import pandas as pd
loc = "../input/nyc-jobs.csv"

df = pd.read_csv(loc)

#df = pd.read_excel(loc)     # if the dataset is excel
df.head()    # Get the first 5 rows 
df.tail(5)
df.info()    # Get a summary of the attributes (Columns) data types and the number of non-null values for each column
df.describe()      # Get a numerical summary for the whole dataset
print(df.shape)  # print both rows number and column number

print(df.shape[0])  # row number

print(df.shape[1])  # col number

print(df.columns)   # Columns names



# you can put columns names into a list by following: 



Col_List = df.columns.tolist()

print(Col_List)
df.rename(columns = {'Job ID':'JobID'})
df['Posting Type'].unique().tolist()
import collections

CounterDic = collections.Counter(list(df['Posting Type']))

print(CounterDic)
df.Agency.value_counts()

X = list(CounterDic.keys())        # or you can write it like that also: df.Agency.value_counts()    # .keys().tolist()

Y = list(CounterDic.values())

print(X)

print(Y)
import seaborn as sns

import matplotlib.pyplot as plt

chart = sns.countplot(df['Posting Type'],label="Counts")  # plot the number of repetition for each data within this header/Col.

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)    # to write the x label 

plt.ylabel('Number of counts')
df.iloc[:,0]   # first col

df.iloc[1,0]   # accessing a certain cell
df = df.drop(['Recruitment Contact'],axis=1)   # Dropping column, should write axis=1 as it indicates to dropping column.
# df = df.drop[1]   # dropping row with index 1 
df = df[df.Level != 'M3']    # if the column contain a data cell named .... drop that row from the dataset

# instead of M3 it could be anyting, like NaN also
import numpy as np

df['Hours'] = df['Hours/Shift']

df['Hours/Shift'].drop

df.Hours = df.Hours.replace(np.nan,0)

print(df.head())

df['WorkLocation'] = df['Work Location 1']

df['Work Location 1'].drop
df.loc[(df.WorkLocation.notnull()),'NewWorkLocation']=1 

df.loc[(df.WorkLocation.isnull()),'NewWorkLocation']=0

print(df.WorkLocation)

print(df.NewWorkLocation)
Xdf = pd.DataFrame(X)     # Remember that X is a list that contain the keys of 'Agency' column

print(Xdf)

df.head(10).to_csv('X head.csv') # here you write the directory
df['Salary Range From'].mean()