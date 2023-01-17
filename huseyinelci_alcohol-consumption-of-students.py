# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# In this code, I read CSV data

data = pd.read_csv('../input/student-alcohol-consumption/student-mat.csv')
# In this code, be shown info()

data.info()
# In this code, be shown correlation map

# corr() with table

data.corr()
# In this code, be shown correlation map

# corr() with color shema

f,ax = plt.subplots(figsize=(21, 7))

sns.heatmap(data.corr(), annot=True, linewidths=.01, fmt= '.2f',ax=ax)

plt.show()
# In this code, be shown expamle inside data

# head() 

data.head(10)
# In this code, be shown columns name of data

# columns() 

data.columns



# It has 33 columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line



data.age.plot(kind = 'line', color = 'g',label = 'age',linewidth=0.5,alpha = 1,grid = True,linestyle = ':')

data.Walc.plot(color = 'r',label = 'Walc',linewidth=0.5, alpha = 1,grid = True,linestyle = '-.')

data.Dalc.plot(color = 'b',label = 'Dalc',linewidth=0.5, alpha = 1,grid = True,linestyle = '-')

plt.legend(loc='center right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label for x axis

plt.ylabel('y axis')              # label = name of label for y axis

plt.title('Age and Dalc&Walc Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = 'Walc' and 'Dalc', y = 'age'

# Example 1 : One table two variables but one color...

data.plot(kind='scatter', x='Walc' and 'Dalc', y='age',alpha = 0.2,color = 'red')

plt.xlabel('Walc')                               # label = name of label for x axis

plt.ylabel('age')                                # label = name of label for y axis

plt.title('Walc - age Scatter Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = 'Walc' and 'Dalc', y = 'age'

# Example 2 : Two table for two different variables

data.plot(kind='scatter', x='Walc', y='age',alpha = 0.2,color = 'DarkBlue', label='Walc')

plt.title('Walc - age Scatter Plot')            # title = title of plot

data.plot(kind='scatter', x='Dalc', y='age',alpha = 0.2,color = 'DarkGreen', label='Dalc')

plt.title('Dalc - age Scatter Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = 'Walc' and 'Dalc', y = 'age'

# Example 3 : One table two variables with different colors...

ax = data.plot(kind='scatter', x='Dalc', y='age',alpha = 0.5,color = 'DarkBlue', label='Dalc')

df = data.plot(kind='scatter', x='Walc', y='age',alpha = 0.2,color = 'Red', label='Walc', ax=ax)

plt.title('Dalc&Walc - Age Scatter Plot')            # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

data.age.plot(kind = 'hist',bins = 14,figsize = (21,5))

plt.xlabel('age') 

plt.title('Age Frequency Histogram') 

plt.show()
# clf() = cleans it up again you can start a fresh

data.age.plot(kind = 'hist',bins = 14)

plt.clf()

# We cannot see plot due to clf()
# Scatter Plot 

# x = 'G1' y = 'G2'

# Example 1 : One table two variables but one color...

data.plot(kind='scatter', x='G1', y='G2',alpha = 0.2,color = 'red')

plt.xlabel('G1')                               # label = name of label for x axis

plt.ylabel('G2')                               # label = name of label for y axis

plt.title('G1 - G2 Scatter Plot')              # title = title of plot

plt.show()
data.columns
# We are computing the average of G1



averageG1 = sum(data.G1) / len(data.G1)

print(averageG1)
# We are writing the Conditional Expression for rows of G1



data["G1_Status"] = ["High" if i > averageG1 else "low" for i in data.G1]
# We are showing G1_Status and G1 for first 20 rows

data.loc[:20,["G1_Status", "G1"]]