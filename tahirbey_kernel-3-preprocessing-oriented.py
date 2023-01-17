# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sns
from collections import Counter 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/turkish-airlines-daily-stock-prices-since-2013/cleanThy.csv')
data.columns
data.columns = ["Date", "Last_Price", "Lowest_Price", "Highest_Price", "Volume"]
#some preprocess on column indexes
data.head(20) #let us have a glance on the data, to have an idea whats going on
data.describe()
data.info()
def plot_variables_byDate(variable):
    #plt.plot(figsize = (9,3))
    #plt.plot(data[variable])
    sns.set(style="whitegrid")
    sns.lineplot(data= data[variable], palette="magma",)
    plt.xlabel('Date')
    plt.ylabel(variable)
    plt.title( '{} vs Date' .format(variable) )
    plt.show()
myVariables = ['Last_Price', 'Lowest_Price', 'Highest_Price', 'Volume']
for i in myVariables:
    plot_variables_byDate(i)
data[['Last_Price','Volume']].groupby(['Volume'],as_index = False).mean()
data[['Lowest_Price','Volume']].groupby(['Volume'],as_index = False).mean()
data[['Highest_Price','Volume']].groupby(['Volume'],as_index = False).mean()
def scatterPlots(variable1,variable2):
    data.plot(kind = 'scatter' , x = variable1, y = variable2, color = 'red', figsize = (15,15))
    plt.xlabel('Lowest_Price')
    plt.ylabel('Highest_Price')
    plt.show()
variable1 = ['Last_Price','Lowest_Price','Highest_Price']
for i in variable1:
    scatterPlots(i,'Volume')
variable1 = ['Last_Price','Lowest_Price']
for i in variable1:
    scatterPlots(i,'Highest_Price')
variable1 = ['Last_Price']
for i in variable1:
    scatterPlots(i,'Lowest_Price')
def detect_outliers(data,features):
    outlier_indices = []
    
    for c in features:
        #1st Quartile
        Q1 = np.percentile(data[c],25)
        #3rd Quartile
        Q3 = np.percentile(data[c],75)
        #IQR
        IQR = Q3 - Q1
        #Outlier Step
        outlier_step = IQR * 1.5
        #detect outliers and indices
        outlier_list_col = data[(data[c] < Q1 - outlier_step)|( data[c] < Q3 + outlier_step )].index
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i,v in outlier_indices.items() if v > 2)
    
    return multiple_outliers
data.loc[detect_outliers(data,['Last_Price','Highest_Price','Lowest_Price','Volume'])]
#Drop Outliers
data.drop(detect_outliers(data,['Last_Price','Highest_Price','Lowest_Price','Volume']),axis = 0).reset_index(drop = True)
myVariables = ['Last_Price', 'Lowest_Price', 'Highest_Price', 'Volume']
for i in myVariables:
    plot_variables_byDate(i)
data.columns[data.isnull().any()]