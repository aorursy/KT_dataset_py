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
import pandas as pd

avocado = pd.read_csv("../input/avocado-prices/avocado.csv")
avocado.tail()
avocado.describe()
avocado_price = avocado["AveragePrice"]

print(avocado_price.value_counts())
import seaborn as sns

sns.boxplot(y="AveragePrice", data=avocado)
# Bar plot of value_counts of different neighborhoods

# Original code from https://python-graph-gallery.com/1-basic-barplot/



import matplotlib.pyplot as plt





# Get data

avocado_price = avocado["AveragePrice"].loc[avocado["AveragePrice"] < 1.2]

counts = avocado_price.value_counts()

categories = avocado_price.value_counts().keys()

y_pos = np.arange(len(categories))



plt.figure(figsize=(90,30))

plt.title("Average Price of Avocados", fontsize=80)

plt.ylabel('Number of Avocados', fontsize=60)

plt.xlabel('Price of Avocados', fontsize=60)



plt.bar(y_pos, counts, color=['red', 'orange', 'yellow', 'lightgreen', 'lightblue', 'cyan', 'magenta', 'purple'])

plt.xticks(y_pos, categories)

plt.show()





# Create bars

#plt.bar(y_pos, counts)

 

# Create names on the x-axis

#plt.xticks(y_pos, categories)

 

# Show graphic

#plt.show()
