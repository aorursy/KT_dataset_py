#First add libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sbn
#Import our CSV data file



kd = pd.read_csv("../input/creditcardfraud/creditcard.csv")
#Take information from data for first impression 



kd.info()
#Take information about columns



kd.columns

#Take some logical information from data here



kd.describe()
#Taking data correlation information here now



kd.corr()
#figure settings for size here. fig is name of figure. ax is settings of figure size for subplots.



fig, ax = plt.subplots(figsize=(20,20))



#Using seaborn for heatmap of correlation

sbn.heatmap(kd.corr(), annot = True , linewidths=.1 , fmt=".1f", ax=ax)

plt.show()
#IDK why we use that here. But I want show that :D Showing first 3 index from top.



kd.head(3)
#Same with up code. But showing from bottom. (default is five index showing if you don't write any number in () part.)



kd.tail()
#Creating scatter plot.



kd.plot(kind="scatter", x="V7", y="Amount",alpha = 0.3,color = "green")

plt.xlabel="V7"

plt.ylabel="Amount"

plt.title("V7/Amount Scatter Plot")

plt.show()
#Creating line plot for data. It's look like too complex but if you find a simple data for line plot, you will say "so easy.."



kd.V2.plot(kind="line",color="blue",linewidth="0.2",figsize=(50,20),alpha=0.8)

plt.title("Line Plot",size=50)

plt.show()
#Create histogram plot. If you don't know or haven't any idea about that plots, you must glance basic math subject "data". It's a word of advice for you from me :)



kd.plot(kind="hist", bins=10 , figsize=(10,10))

plt.show()
#It's simple filter using for data. If you look the compare of datas in output, you can see the filter with "false" and "true". If you don't know that you must glance sucbect of python "booleans".



a = kd["Amount"]

print(a)

print("-------------------------------------------------------\n")

print(a>100)
#You can multiple filter with "&" operator.



kd[(kd.V7 > 1.5) & (kd.V8 > 1.5)]
#Giving selected columns data info.



kd.loc[:,"V8"]