# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns  # data visualization

import matplotlib.pyplot as plt # create static , animated

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read the Data File 

my_data = pd.read_csv("/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv")

my_data.info()  # See the Basic Information About Our Data 
# See the Type of Our Data . Is it series or dataframe ?



type(my_data)
# Count of Rows and Columns



my_data.shape
# Get the Columns of Our Data



my_data.columns

my_data.index
# Top 10 Rows of  Data



my_data.head(10)

# Last 10 Rows of Data



my_data.tail(10)
# Get the Rows Increasing in a Range



my_data[10:100:5]  # [start : finish : step]
# Summarize Statistcs of Our Data



my_data.describe()
# Correlation Map of Data



my_data.corr()
# Visualization the Correlation Map 



f,ax = plt.subplots(figsize = (20,20))

shape = np.triu(my_data.corr())

sns.heatmap(my_data.corr() , annot = True , linewidths = .5 , square = True , mask = shape , fmt = '.4f' , cmap = "inferno" , ax = ax , linecolor = "grey" )

plt.show()



# annot = visibility of number 



# cmap = color of correlation map 



# mask = change the matrix shape (it must be : .triu() or tril()  )

# Filtering DataFrame



my_data[(my_data['year'] == 1998) & (my_data['sex'] == 'female') & (my_data['age'] == '15-24 years') & (my_data['population'] > 300000)]
# I changed the column name because it give Syntax Error when ı use Line Plot .



my_data.rename(columns = {'suicides/100k pop':'suicides'} , inplace = True )

                 

my_data.columns
# Line Plot





plt.figure(figsize=(30,30))

plt.plot( my_data.year , my_data.suicides , color="k" , lw = 2 , ls = 'dotted') # lw = linewidth # ls = linestyle

plt.title("Line Plot") 

plt.xlabel("Year")

plt.ylabel("Age")

plt.show()
# Scatter Plot  



# Analyze the Correlation Between Population and Suicides



my_data.plot(kind = 'scatter' , x = 'population' , y = 'suicides' , marker='s' , figsize = (10,10) , c = 'rosybrown')  # s = square c = color

plt.title("Correlation Between Population and Suicides ")

plt.show()

# Histogram Plot



# Analyze the Age Distribution 



# my_data.age = pd.to_numeric(my_data.age)

# my_data.age.plot(kind='hist')

# plt.show()
# Histogram Plot



# sns.barplot(x = my_data.groupby('age')['sex'].count().index , y = my_data.groupby('age')['sex'].count().values)

# plt.xticks(rotation=30)

# plt.show() 
# temp = re.findall(r'\d+', my_data['age']) 

# res = list(map(int, temp)) 



# I extracted numerical values to list from age column 



col_one_list = my_data['age'].tolist()

print(col_one_list)

# Analyze the Age Distribution With Histogram Plot



plt.figure(figsize=[10,10])

plt.hist(col_one_list , color = 'gray' , bins = 10)

plt.title("Analyze the Age Distribution")

plt.show()