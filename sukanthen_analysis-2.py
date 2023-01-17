import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 
data=pd.read_excel('../input/covid19-india/Complete COVID-19_Report in India.xlsx')

data.head()
data.columns
data['Date Announced'].value_counts()
x = data['Date Announced'].value_counts()
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

plt.plot(x)

plt.show()
sns.set()
gender_info = data['Gender'].value_counts()

gender_info.plot(kind='bar') #Lots of missing values.
x1 = data['Current Status'].value_counts()

x1.plot(kind='bar')
x1 #24 Deceased or Dead
x2=data['Nationality'].value_counts() #Affected patients in India 

x2.plot(kind='bar')
x2
df = data

plot_df = df.dropna()

plt.hist(plot_df['Age Bracket'])
plot_df.head()
plot_df.shape