import csv

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mtplt

import seaborn as sns
data = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')

data
data.head()
data.tail()
print(data.shape)

print(type(data))
df1 = data.set_index('Platform')

df1
df2 = df1.groupby('Platform').sum()

df2
df3 = df2.iloc[:,1:6]

df3
df4 = df3.sort_values('Global_Sales',ascending=False)

df5 = df4.head(15)

df5
df5.plot(kind='bar', stacked=True, colormap = "cool")



plt.show()
df5.plot(kind = "pie", y = "Global_Sales", legend = False, colormap ="CMRmap")



plt.show()
df5.plot.area(stacked=False)



plt.show()
PS3_Sales = df1.loc["PS3"]

PS3_Sales = PS3_Sales[['Year_of_Release','Global_Sales']]

PS3_Sales = PS3_Sales.groupby('Year_of_Release').sum()



X360_Sales = df1.loc["X360"]

X360_Sales = X360_Sales[['Year_of_Release','Global_Sales']]

X360_Sales = X360_Sales.groupby('Year_of_Release').sum()



Wii_Sales = df1.loc["Wii"]

Wii_Sales = Wii_Sales[['Year_of_Release','Global_Sales']]

Wii_Sales = Wii_Sales.groupby('Year_of_Release').sum()



PS3_Sales.plot(kind='bar', color = "b", title = "PS3 Sales by Year")

X360_Sales.plot(kind='bar', color = "g", title = "Xbox 360 Sales by Year")

Wii_Sales.plot(kind='bar', color = "r", title = "Wii Sales by Year")



plt.tight_layout()

plt.show()
df1['User_Score'] = df1['User_Score'].convert_objects(convert_numeric= True)



sns.jointplot(x='Critic_Score',y='User_Score',data=df1 ,kind='hex')



plt.show()
sns.regplot(x="Critic_Score", y="Global_Sales", data=df1, color = 'green')



plt.show()
sns.regplot(x="User_Score", y="Global_Sales", data=df1, color = 'green')



plt.show()