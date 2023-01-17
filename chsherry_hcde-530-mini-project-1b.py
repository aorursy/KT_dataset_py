# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/unemployment/unemploymentdata.csv")
data.head(5)

#Overview 

#As I majored in economics during undergrad, I wanted to learn how to plot basic employment data charts as I have never done that before. 
#I wanted to plot the unemployment data, as well as a visual representation of the percentage of argriculture industry as employment over the years. 

#Data profile
#I'm interested in researching past unemployment in the US. This is the data that I took from the US Bureau of Labour Statistics. 

lines = data.plot.line(x='year', y='unemployed_percent')

#Analysis 
#This is a line graph of unemployment data since 1940. As you can see, unemployment increased during the dot com bubble and during the great recession to 10%. 
#It's astouding to think that the current unemployment in 2020 may reach 25%, given that in the most recent 80 years, the highest has been only 10%. 

lines = data.plot.line(x='year', y='population')
lines = data.plot.line(x='year', y='agrictulture_ratio')

#Analysis 
#As the population in the US grows, we can compare that to the ratio of employed population that work in agriculture.
#The argiculture employment ratio has decreased drastically. This could be attributed to globalization, the rise in machinery, and shift from argricultured base economy to services.

import matplotlib.pyplot as plt

plt.plot( 'year', 'population', data=data)
plt.plot( 'year', 'unemployed', data=data)
plt.plot( 'year', 'employed_total', data=data)

#Analysis
#This is a summary graph of US population, total employment and unemployment since 1940. 
#The blue line represents the population growth, the green line represents the total employed population, while the orange line represents the total unemployed. 

#Conclusion
#By learning the basics of python and data visualization, I feel excited to be able to plot these basic graphs that I've always seen in textbooks. 
#For future exploration, I would like to analyze each state's employment data and percentage of workforce in agriculture. 
#This can tell me which states over time have experienced faster industrialization. 
#I can compare that data to each states unemployment data to extrapoliate the effects of industralization on employment across the US. 