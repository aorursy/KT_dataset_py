# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import time
from matplotlib import animation
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('/kaggle/input/world-population-19602018/population_total_long.csv')
dataset = dataset.groupby(['Year']).apply(lambda x: x.sort_values(["Count"],ascending=False)).reset_index(level=1,drop=True).drop(columns=['Year'])
indexes = list(set(dataset.index))

for i in indexes:
    

    serie = dataset.at[i , 'Country Name']
    serie = serie[0:20]
       # print(len(serie))
    pop = dataset.at[i, 'Count']
    pop = pop[0:20]
    plt.figure(num = None, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')
    fig = plt.figure(figsize=(16,8))
    plt.bar(serie, pop, color = ['r','g','b'], width = 0.50)
    plt.title("  Year \n " + str(i))
    plt.xlabel("Country")
    plt.ylabel("Population")
    plt.xticks(rotation = 90 , fontsize = 15)
    plt.show()
    time.sleep(1) 
    #print(pop)
    #plt.yticks(rotation=90)
    #plt.savefig("sample.jpg")
  
