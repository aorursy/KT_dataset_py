import numpy as np

import pandas as pd

import matplotlib.pyplot as ply

import seaborn as sns



df = pd.read_csv('/kaggle/input/traincsv/train-200907-141856.csv')
sns.countplot(x='Survived', data=df) #Count_Plot

sns.countplot(x='Survived', hue='Sex',data=df) #Count_Plot
sns.lineplot(x='Age',y='Fare',data=df) #Line_Chart
df.plot.scatter(x='Age', y='Fare')  #Scatter_Plot
df.plot.scatter(x='Age', y='Survived') #Scatter_Plot
df['Age'].hist(bins=70)   #histogram
import matplotlib.pyplot as plt

sizes= df['Survived'].value_counts()

fig1,ax1 = plt.subplots()

ax1.pie(sizes,labels=['Not Survived',

'Survived'],autopct='%1.1f%%',shadow=True)

plt.show()                                    #Pie_Chart