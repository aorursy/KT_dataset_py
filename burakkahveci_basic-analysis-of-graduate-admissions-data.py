# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data= pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
data = data.rename(columns={'GRE Score': 'GRE_Score', 'TOEFL Score': 'TOEFL_Score','University Rating': 'University_Rating',

                           'Chance of Admit ': 'Chance_of_Admit'})
data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(15)
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.CGPA.plot(kind = 'line', color = 'g',label = 'CGPA',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Chance_of_Admit.plot(color = 'r',label = 'Chance_of_Admit',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
data.plot(kind='scatter', x='GRE_Score', y='Chance_of_Admit',alpha = 0.5,color = 'red')

plt.xlabel('GRE_Score')              # label = name of label

plt.ylabel('Chance_of_Admit')

plt.title('Scatter Plot')   

plt.show()
data.plot(kind='scatter', x='TOEFL_Score', y='Chance_of_Admit',alpha = 0.5,color = 'blue')

plt.xlabel('TOEFL_Score')              # label = name of label

plt.ylabel('Chance_of_Admit')

plt.title('Scatter Plot')   

plt.show()
# Histogram

# bins = number of bar in figure

data.TOEFL_Score.plot(kind = 'hist',bins = 70,figsize = (12,12))

plt.title('Histogram')  

plt.show()
# Histogram

# bins = number of bar in figure

data.GRE_Score.plot(kind = 'hist',bins = 105,color='r',figsize = (12,12))

plt.title('Histogram')  

plt.show()
# Histogram

# bins = number of bar in figure

data.University_Rating.plot(kind = 'hist',bins = 50,color='black',figsize = (12,12))

plt.title('Histogram')  

plt.show()