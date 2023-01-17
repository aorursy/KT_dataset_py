import pandas as pd

df = pd.read_csv("../input/vgsales.csv")

df.head()

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

#checking the relationship between the numerical variables

df=df.select_dtypes(include=['float64','int64'])

corr=df.corr()

f,ax=plt.subplots(figsize=(11,9))

cmap=sns.diverging_palette(220,10,as_cmap=True)

sns.heatmap(corr,cmap=cmap,vmax=1,square=True,linewidths=.5,cbar_kws={"shrink": .5},ax=ax)







#first degree regression

p1=np.polyfit(df.NA_Sales,df.Global_Sales,1)

#drawing scatter plot between NA_sales and Global sales

plt.plot(df.NA_Sales,df.Global_Sales,'o')

plt.plot(df.NA_Sales,np.polyval(p1,df.NA_Sales),'g')

plt.xlabel('Na_Sales')

plt.ylabel('Global_Sales')

plt.title('regression line between na sales and global sales')

plt.plot(df.NA_Sales,df.Global_Sales,'o')

xp=np.linspace(-30,70,5)

plt.plot(xp,np.polyval(p1,xp),'y')

plt.xlabel('Na_Sales')

plt.ylabel('Global_Sales')

plt.title('Perdiction with one variable')

yfit=p1[0]*df.NA_Sales+p1[1]

print(yfit)

#these are the values of Global sales for na_Sals regression line