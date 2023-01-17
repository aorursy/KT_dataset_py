# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/world-happiness-report-2019/world-happiness-report-2019.csv")
data.head()
data.info()
#renaming the columns

data= data.rename(columns={"Country (region)":"Country","Log of GDP\nper capita":"Per capita","Healthy life\nexpectancy":"Life expectancy"})
data.head()
#checking the null values

data.isnull().sum()
data.describe()
#looking at the distribution of the below columns

import seaborn as sns

for col in ["Positive affect","Negative affect","Social support","Freedom","Corruption","Generosity","Per capita","Life expectancy"]:

    sns.distplot(data[col])

    plt.show()
#Imputing the missing values with mean

for col in ["Positive affect","Negative affect","Social support","Freedom","Corruption","Generosity","Per capita","Life expectancy"]:

    mean_value = data[col].mean()

    data[col]= data[col].fillna(mean_value)
data.isnull().sum()
#Scatter plots between ladder and other variables

import matplotlib.pyplot as plt

for col in ["Positive affect","Negative affect","Social support","Freedom","Corruption","Generosity","Per capita","Life expectancy","SD of Ladder"]:

    data.plot(kind = 'scatter',

          x = col,

          y = 'Ladder')

    plt.show()
# Drawing the regression plots

import seaborn as sns

sns.regplot(x='Life expectancy', y = 'Ladder', data = data)
# Drawing the regression plots

import seaborn as sns

sns.regplot(x='Corruption', y = 'Ladder', data = data)
# Drawing the regression plots

import seaborn as sns

sns.regplot(x='Freedom', y = 'Ladder', data = data)
# Drawing the regression plots

import seaborn as sns

sns.regplot(x='Social support', y = 'Ladder', data = data)
# Drawing the regression plots

import seaborn as sns

sns.regplot(x='Per capita', y = 'Ladder', data = data)
data.columns
# Drawing the regression plots

import seaborn as sns

sns.regplot(x='Positive affect', y = 'Ladder', data = data)
# Drawing the regression plots

import seaborn as sns

sns.regplot(x='Negative affect', y = 'Ladder', data = data)
# Drawing the regression plots

import seaborn as sns

sns.regplot(x='SD of Ladder', y = 'Ladder', data = data)
# Drawing the regression plots

import seaborn as sns

sns.regplot(x='Generosity', y = 'Ladder', data = data, color= 'red', marker = '+')
data.columns
data_numeric = data[["Ladder","SD of Ladder","Positive affect","Negative affect","Social support","Freedom","Corruption","Generosity","Per capita","Life expectancy"]]
# Correlation matrix

cor = data_numeric.corr()

cor
# Plot the correlation on a heat map

# Figure size

plt.figure(figsize=(16,8))



# Heatmap

sns.heatmap(cor, cmap="YlGnBu", annot=True)

plt.show()