# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataUsers = pd.read_csv('../input/Users.csv') 
dataUsers.head()   # it shows us just first 5 rows of our dataset
dataUsers.tail()  # it shows us just last 5 rows of our dataset
dataUsers.info()   # info about dataset, for exp. there are 1933837 entries in this dataset
dataUsers.columns # if we want to see just columns name
dataFrameUsers = dataUsers.copy()
# I know, this is long way to find out how many users join in kaggle according to years. 
# But i think, it can be a good way just for practice about python syntax :)
dataRegister1 = dataFrameUsers[['RegisterDate']]
dataRegister1
count10 = 0
count11 = 0
count12 = 0
count13 = 0
count14 = 0
count15 = 0
count16 = 0
count17 = 0
count18 = 0

for each in dataRegister1["RegisterDate"]:
    if ("2010" in each):
         count10 += 1
    elif("2011" in each):
         count11 += 1
    elif("2012" in each):
         count12 += 1
    elif("2013" in each):
         count13 += 1
    elif("2014" in each):
         count14 += 1
    elif("2015" in each):
         count15 += 1
    elif("2016" in each):
         count16 += 1
    elif("2017" in each):
        count17 += 1
    elif("2018" in each):
        count18 += 1
print("2010:",count10," 2011:",count11," 2012:",count12," 2013:",count13," 2014:",count14," 2015:",count15," 2016:",count16," 2017:",count17," 2018:",count18)

# to find out years one by one 
def yilSay(x):
    y = str(x)
    count = 0
    for each in dataUsers['RegisterDate']:
        if (y in each):
             count = count + 1
    return count

print(yilSay(2017))  #  #user joined Kaggle in 2017
# to find out all of the years results
def f(*args):   
    for years in args:
        print(yilSay(years))

result = f(2010,2011,2012,2013,2014,2015,2016,2017,2018)
print(type(result))
# and the last one is for if you want to find out all of the years results and put it into a list
countUsersList = list()
def f(*args):   
    for years in args:
        countUsersList.append(yilSay(years))

resultYears = f(2010,2011,2012,2013,2014,2015,2016,2017,2018)
print(countUsersList)

# create a list for new dataframes index column
i = 0
liste = list()
while i != 9 :
    liste.append(i)
    i += 1

#create dataframe
idx = liste
colNames =  ['Years', 'UserCounts']
df  = pd.DataFrame(index = idx, columns = colNames)
df
# create years
i = 2010
listeYears = list()
while i != 2019 :
    listeYears.append(i)
    i += 1
print(listeYears)

# filling columns
df['Years'] = listeYears
df['UserCounts'] = countUsersList
df
# plotting with Scatter Plot
df.plot(kind='scatter', x='Years', y='UserCounts', color = 'red')
plt.xlabel = 'Years'
plt.ylabel = 'UserCounts'
plt.title = '#Users by Years'
plt.show()
# again scatter plot
plt.scatter(df.Years,df.UserCounts,color = 'red', alpha = 0.5)
plt.show()
# Line plot
df.loc[:,["Years","UserCounts"]]
df.plot()
plt.show()
df.plot(subplots = True)
plt.show()
# if we want to show the performance tiers which are higher than 4
x = dataUsers['PerformanceTier'] > 4
dataUsers[x]
#filtering pandas for logical and
dataUsers[np.logical_and(dataUsers['Id'] > 1000000 , dataUsers['PerformanceTier'] >= 3)]
# if we want to count
data = dataUsers[np.logical_and(dataUsers['Id'] > 1000000 , dataUsers['PerformanceTier'] >= 3)]
count_row = data.shape[0]
print("Count Row : ",count_row)
# same result with 18th line
dataUsers[(dataUsers['Id'] > 1000000) & (dataUsers['PerformanceTier'] >= 3)] 
# or we can use like this, again same result 18th line
first_filter = dataUsers.Id > 1000000
second_filter = dataUsers.PerformanceTier >= 3
dataUsers[first_filter & second_filter]
# add new column to dataframe and use list comprehension
dataFrameUsers["PerformanceTierLevel"] = ["HighLevel" if i > 3 else "LowLevel" for i in dataFrameUsers.PerformanceTier]
dataFrameUsers
# new dataset - we have NaN values
dataKernels = pd.read_csv('../input/Kernels.csv') 
dataKernels.info()   # info about dataset
dataKernels.head()
dataKernels.shape    # dataset has 183797 rows and 16 features
dataKernels.describe()   # ignore null entries 
# if there are nan values that also be counted
print(dataKernels.ForumTopicId.value_counts(dropna = False )) # have 170307 NaN values
dataMissing = dataKernels.loc[0:10,"Id":"CreationDate"]
# if we want to drop NaN values and save df 
dataMissing['ForumTopicId'].dropna(inplace = True) 
# inplace = True means save df without NaN values
dataMissing
# take a look our df, you can notice that NaN values still remain. 
# Unfortunately I don't know why :( if you know please inform me. But anyway I'll practise a few steps later again :)
# go back line 29 and check it is true or false
assert dataMissing["ForumTopicId"].notnull().all()
# return nothing because we drop NaN values
# filling values 0 insted of NaN
dataMissing["ForumTopicId"].fillna('0', inplace=True)
dataMissing
# Now creating new dataframe in orter to see dropna, fillna results
dataF = {'country': ['Turkey','Sweden' ,np.nan, 'Spain', 'Italy'], 
            'capital': ['Ankara', 'Stockholm','Helsinki', 'Madrid', 'Roma'], 
            'population': [5445, np.nan, np.nan, 3166 ,2868]}
dF = pd.DataFrame(dataF, columns = ['country', 'capital', 'population'])
dF
dF.dropna()  # NaN values are gone, but df have same values
dF['country'].dropna()  # NaN values are just gone for country columns but df still same
dF2 = dF.copy()
dF2["population"].fillna('0', inplace=True)  # filling values 0 insted of NaN and save
dF2
# if we want to permanently eliminate NaN values, use inplace = True
dF3 = dF.copy()
dF3.dropna(inplace=True)
dF3
# if we want to combine 2 dataframes
data1 = dataKernels.head()
data2 = dataKernels.tail()
conc_data_row = pd.concat([data1,data2], axis=0, ignore_index = True)  # axis = 0 : adds dataframes in row
conc_data_row
# if we want to find out type of data4
data2.dtypes
# Time Series
dataTL = dataKernels.loc[0:30,"Id":"TotalVotes"]
dataTimeList = dataKernels["CreationDate"].loc[0:30]
dataTimeList
#dataTimeList['CreationDate'] = pd.to_datetime(dataKernels['CreationDate'])
datetime_object = pd.to_datetime(dataTimeList.tolist())
print(type(datetime_object))   
dataTL["DateCreation"] = datetime_object.to_datetime()
dataTL.set_index("DateCreation")  #  dataTL is timeseries
dataKernels = pd.read_csv('../input/Kernels.csv') 
# Hierarchical Index
dataKernels1 = dataKernels.set_index(["AuthorUserId","TotalVotes"])  
dataKernels1.head(5) 
# take columns from TotalViews to TotalVotes and rows up to 30
dataKernels1 = dataKernels.loc[0:20,"TotalViews":"TotalVotes"]
dataKernels1
dataKernels.groupby("AuthorUserId").TotalViews.max()
