# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.
ksdata = pd.read_csv("../input/2015.csv")
ksdata.drop(["Happiness Rank"],axis=1,inplace = True)
ksdata.drop(["Dystopia Residual"],axis=1,inplace = True)

ksdata.rename(columns={'Economy (GDP per Capita)': 'Economy'}, inplace=True)
ksdata.rename(columns={'Health (Life Expectancy)': 'Health'}, inplace=True)
ksdata.rename(columns={'Trust (Government Corruption)': 'Trust'}, inplace=True)
ksdata.rename(columns={'Happiness Score': 'Happiness'}, inplace=True)
ksdata.head()
ksdata.tail()
ksdata.info()
ksdata.columns
ksdata.describe()
ksdata[ksdata.Country == "Turkey"]
AVG_ECONOMY = ksdata.Economy.mean()
ksdata["wealth"]=["POOR" if AVG_ECONOMY> each else "RICH" for each in ksdata.Economy]
AVG_FAMILY = ksdata.Economy.mean()
ksdata["family ties"]=["weak" if AVG_FAMILY> each else "powerful" for each in ksdata.Economy]
ksdata[70:80]

#kksdata = ksdata['Family'].rank(method='min')

ksdata.plot(kind='scatter', x='Economy', y='Health',alpha = 0.5,color = 'red')
plt.xlabel('Economy')              
plt.ylabel('Health')
plt.title('Economy / Health')
plt.show()
data1 = ksdata.loc[:,["Family","Happiness"]]
data1.plot()
plt.show()
plt.hist(ksdata.Freedom,bins=50)
plt.xlabel("PetalLengthCm values")
plt.ylabel("frekans")
plt.title("hist")
plt.show
plt.bar(ksdata.Economy,ksdata.Health)
plt.title("bar plot")
plt.xlabel("Economy")
plt.ylabel("Health")
plt.show()