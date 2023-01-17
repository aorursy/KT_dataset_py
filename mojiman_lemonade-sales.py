# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print('Perawat Wongkrit ID: 6010603')
# Import all needed library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
# Upload a csv file 
dataset= pd.read_csv("../input/Lemonade.csv")
# Find Average Sales
sale_mean = round(dataset["Sales"].mean(),2)
print("Average Sales of Lemonade is ",sale_mean)
# Show all records that sale below average sale
records = dataset[dataset['Sales'] < sale_mean]
print(records)
# Show a scatter plot of sales and temperature
import matplotlib.pyplot
import pylab

tem_list = dataset["Temperature"].tolist()
sales_list = dataset["Sales"].tolist()

matplotlib.pyplot.scatter(tem_list,sales_list)

matplotlib.pyplot.show()
# Show average sales on each day
saleMonday = round(dataset[dataset['Day']=='Monday']['Sales'].mean(),2)
saleTuesday = round(dataset[dataset['Day']=='Tuesday']['Sales'].mean(),2)
saleWednesday = round(dataset[dataset['Day']=='Wednesday']['Sales'].mean(),2)
saleThursday = round(dataset[dataset['Day']=='Thursday']['Sales'].mean(),2)
saleFriday = round(dataset[dataset['Day']=='Friday']['Sales'].mean(),2)
saleSaturday = round(dataset[dataset['Day']=='Saturday']['Sales'].mean(),2)
saleSunday = round(dataset[dataset['Day']=='Sunday']['Sales'].mean(),2)
raw_data = {'Avr. Sale': [saleMonday,saleTuesday,saleWednesday,saleThursday,saleFriday,saleSaturday,saleSunday]}
df = pd.DataFrame(raw_data, index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
df
