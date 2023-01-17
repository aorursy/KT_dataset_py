import seaborn as sns

from sklearn import preprocessing

import pandas as pd 

import numpy as np

import matplotlib

import matplotlib.pyplot as plt  

matplotlib.style.use('ggplot')

%matplotlib inline

import math

import matplotlib as mpl

import plotly



input_df = pd.read_csv("../input/appendix.csv",sep=',',parse_dates=['Launch Date'])

input_df['year'] = input_df['Launch Date'].dt.year

print(input_df.columns)
df = pd.read_csv("../input/appendix.csv")

df.describe()
df.dtypes
labels = 'Computer Science', 'Government, Health, and Social Science', 'Humanities, History, Design, Relgion, and Education', 'Science, Technology, Engineering, and Mathematics'

sizes = [10, 26, 32, 31]

colors = ['yellow', 'skyblue', 'green', 'red']



plt.pie(sizes,               

        labels=labels,      

        colors=colors,      

        autopct='%1.1f%%',  

        startangle=30       

        )



plt.axis('equal')



plt.show()
correlation = df.corr()

plt.figure(figsize=(10,10))

sns.heatmap(correlation, vmax=1, square=True,annot=True)



plt.title('Correlation between different fearures')
df.corr()
df.mean()