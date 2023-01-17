# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
import pandas.io.date_converters as conv


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/athlete_events.csv')

# Gives general information about data. Data type, columns name,
# Total columns, null object or non-null object, 
# Number of Index, File size gives information such as
data.info()
#data.head() First 5 lines by default

data.head(10)
#data.tail() Last 5 lines by default

data.tail(10)
# Gives the data types and column names of the columns

data.dtypes
data['City'] #Returns the 'City' column in 'Data'.With ID
data.columns #Returns The columns of the 'Data'.
data.corr() # correlation between features
#CORRELATION MAP
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

#annot=True :It gives us correlation values inside the boxes
#linewidths= .5 :Thickness of line between boxes
#fmt= '.1f' :It gives how many will be written of correlation values after comma

#Line Plot 
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Age.plot(kind='line', color='blue',label='Age',linewidth=1,alpha=0.5,grid=True,linestyle='-')
data.Height.plot(kind='line',color = 'yellow',label ='Height',linewidth=1, alpha = 0.5,grid = True,linestyle = '-')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Age/Height')
plt.show()
#Scatter is better when there is correlation between two variables
data.plot(kind='scatter',x='Age', y='Year',alpha=0.5,color='red')
plt.xlabel('Age')
plt.ylabel('Year')
plt.title('Age-Year Scatter Plot')
plt.show()
#Histogram is better when we need to see distribution of numerical data.
#bins = number of bar in figure
data.Height.plot(kind='hist',bins=50,figsize=(18,18),grid=True)
plt.xlabel('Height')
#plt.clf()
plt.show()
x = data['Age']>40
data[x]
#Returns over 40 years old

# Filtering Pandas with logicial_and method
x = np.logical_and(data['Age']>24,data['Height']>170)
data[x]
#Age greater than 24 and greater than 170 cm
x = np.logical_and(data['Sex'] == 'F',data['Year']  >= 1992) #women of gender and older than olympic year 1992
y = np.logical_and(data['Age']>20,data['Height']>170)  #Age greater than 20 and greater than 170 cm
z = np.logical_and(x,y) #'Sex' == 'F', 'Year' >= 1992 , 'Age' > 20, 'Height' > 170cm
#data[z] 
a = np.logical_and(data['NOC'] == 'USA',data['Season'] == 'Summer') # 'NOC'== 'USA', 'Season' = 'Summer'
b = np.logical_and(data['Medal'] == 'Gold',data['City'] == 'London') # 'Medal' = 'Gold', 'City' = 'London'
c = np.logical_and(a,b) #'NOC'== 'USA', 'Season' = 'Summer', 'Medal' = 'Gold', 'City' = 'London'
#data[c]

d = np.logical_and(z,c) #'Sex' == 'F', 'Year' >= 1992 , 'Age' > 20, 'Height' > 170cm, 'NOC'== 'USA', 'Season' = 'Summer', 'Medal' = 'Gold', 'City' = 'London'
data[d]
data.describe() #Just numerical datas

data.sort_index(axis=1, ascending=False) #Sorting by an axis

data.sort_values(by='Age') #Sort by values
data[0:3]
data[1900:1993] #for index
data.iloc[1900] #1900. index
data.iloc[3:5,0:2] #3 to 5 index and 0 to 2nd column
data.iloc[[1,2,4],[0,2]] #index number 1,2,4 and column number 0 and 2
data.iloc[1:3,:] #1 to 3 index and all columns
dates = pd.date_range('20180101', periods=6)
dates
dataFrame = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
dataFrame
