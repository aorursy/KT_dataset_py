# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

import math

%matplotlib inline
# Reading the dataset file

df = pd.read_csv('../input/2016.csv')
# Displaying the first 5 rows of the DataFrame

df.head()
# Data type in each column

df.dtypes
# Summary of information in all columns

df.describe()
# Number of rows and columns in the DataFrame

df.shape
# Which country has the highest Happiness Score?

df[df['Happiness Score'] == 7.526 ]
# Which country has the lowest Happiness Score?

df[df['Happiness Score'] == 2.905 ]
# Checking the Africa region

region_africa = df[df['Region'] == 'Sub-Saharan Africa']

region_africa.head()
# Analyzing the Happinness Score the Africa region

values_africa = region_africa.groupby('Country')['Happiness Score'].sum()

values_africa.plot(kind='line', rot=45, color='red')



plt.show()
#Analyzing the Happiness Score of all countries in 2016

sns.swarmplot(x="Region", y="Happiness Score",  data=df)

plt.xticks(rotation=90)



plt.show()
# Checking the position of New Zealand

df[df['Country'] == 'New Zealand']
# Checking the position of Brazil

df[df['Country'] == 'Brazil']
# Joining in a DataFrame the countries of the region of Latin America and Oceania

latin_america = df[df.Region=='Latin America and Caribbean']

oceania = df[df.Region=='Australia and New Zealand']

compare_two = pd.concat([latin_america,oceania],axis=0)

compare_two.head()
# Comparing the Economy with the Happiness Score of the region of Latin America and Oceania

sns.lmplot(data=compare_two,x='Economy (GDP per Capita)',y='Happiness Score',hue="Region")



plt.show()