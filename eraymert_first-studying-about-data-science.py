# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # visualization tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/2017.csv')

data.info()

data.corr() # correlation between features
f,ax = plt.subplots(figsize = (16,16))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()
data.head(10) # This coding gives us first 10 rows.
data.columns # this coding gives us name of columns
# This coding changes columns name
data.rename(columns={"Happiness.Rank": "Happiness_Rank", "Happiness.Score": "Happiness_Score",
                    "Whisker.high":"Whisker_high", "Whisker.low":"Whisker_low", "Economy..GDP.per.Capita.":"Economy_GDP_per_Capita",
                    "Health..Life.Expectancy.":"Health_Life_Expectancy","Trust..Government.Corruption.":"Trust_Government_Corruption","Dystopia.Residual":"Dystopia_Residual"}, inplace = True)
data.columns
data.Economy_GDP_per_Capita.plot(kind='line', color='g', label='Economy_GDP_per_Capita', linewidth=1, 
                          alpha=0.5, grid=True, linestyle=':' )
data.Family.plot(color= 'r', label='Family', linewidth=1, 
                          alpha=0.5, grid=True, linestyle='-.' )
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()

data.plot(kind="scatter", x="Economy_GDP_per_Capita", y="Family", alpha=0.5,
          color="red")
plt.xlabel("Economy_GDP_per_Capita")
plt.ylabel("Family")
plt.title("Economy_GDP_per_Capita Family Scatter Plot")
plt.show()
data.Economy_GDP_per_Capita.plot(kind="hist", bins=50, figsize=(12,12))
plt.show()
#create dictionary and look its keys and values
dictionary = {"Eray": "24", "Ali":"40", "Samet": "20"}
print(dictionary.keys())
print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique

dictionary['Eray']="23"
print(dictionary)
dictionary['Mert'] = "32"
print(dictionary)
del dictionary['Eray']
print(dictionary)
print('Eray' in dictionary)
