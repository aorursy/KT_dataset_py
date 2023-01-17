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
US_Accidents = pd.read_csv("../input/us-accidents/US_Accidents_Dec19.csv")

US_Accidents.head()
accidents = US_Accidents["State"].value_counts()

print(accidents)
import matplotlib.pyplot as plt



#bars=accidents["State"]

#height=accidents.value_counts()



#accidents = US_Accidents.loc[US_Accidents["State"] == "CA" or "TX" or 

#height = accidents.value_counts()

#bars = height.keys()



# Make a fake dataset:

height = [663204,298062,223746,146689,142460,137799,90395,88694,86390,83620,79957,70840,62727]

bars = ('CA', 'TX', 'FL', 'SC', 'NC', 'NY', 'PA', 'MI', 'IL', 'GA', 'VA', 'OR', 'MN')

y_pos = np.arange(len(bars))



# Create bars

plt.bar(y_pos, height)



# Create names on the x-axis

plt.xticks(y_pos, bars)



# Show graphic

plt.show()
import matplotlib.pyplot as plt

#import seaborn as sns



graph = US_Accidents["State"].value_counts().plot.bar(

    figsize=(26, 12), 

    fontsize = 20,

    color = 'mediumpurple'

)

graph.set_title('Number of US Accidents by State', fontsize=40)



#graph = sns.barplot(palette = 'rocket')



#plt.figure(figsize=(26, 12))



#plt.figure(fontsize=40)



#accidents = US_Accidents.loc[US_Accidents["State"] == "CA" or "TX" or 

#height = accidents.value_counts()

#bars = height.keys()



# Make a fake dataset:

plt.ylabel("# of Accidents", fontsize = 20)

plt.xlabel("States", fontsize = 20)



y_pos = np.arange(len(bars))



# Create bars

#plt.bar(y_pos, height)



# Create names on the x-axis

#plt.xticks(y_pos, bars)



# Show graphic

plt.show()