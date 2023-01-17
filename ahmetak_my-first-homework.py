# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/50_Startups.csv")
data.info()
data.head(10)

data.shape
data.describe()
data.corr
f,ax = plt.subplots(figsize=(6, 6))
sns.heatmap(data.corr(),annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()

data.columns
update=['rd_spend','administration','marketing_spend','state','profit']
data.columns=update
data.columns
#line plot 
data.administration.plot(kind = 'line', color = 'g',label = 'Administration',linewidth=1,alpha = 1,grid = True,linestyle = ':' )
data.profit.plot(color = 'r',label = 'Profit',linewidth=1, alpha = 1,grid = True,linestyle = '-.')     
data.rd_spend.plot(kind = 'line', color = 'b',label = 'rd_spend',linewidth=1,alpha = 1,grid = True )    
data.marketing_spend.plot(kind = 'line', color = 'y',label = 'marketing_spend',linewidth=1,alpha = 1,grid = True ) 
plt.legend(loc='upper right')     
plt.xlabel('x axis')              
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()
#Histogram plot
data.profit.plot(kind = 'hist',bins = 10,figsize = (10,6), color = 'r', alpha=0.8)
plt.show()
sns.catplot(x="rd_spend", y="marketing_spend", hue="profit",
            kind="swarm", data=data);
x=data['profit']>190000
data[x]
data[np.logical_and(data['profit']>190000, data['marketing_spend']>400000 )]
data.head()
data[(data['profit']>190000) & (data['marketing_spend']>400000) & (data['administration']>100000)]
def f(*args):
    for i in args:
        print (i)
a=data.rd_spend
b=data.administration
c=data.marketing_spend
f(a,b,c)
data['total_cost'] = data.rd_spend + data.administration +data.marketing_spend
data['total'] = data.total_cost + data.profit
threshold = sum(data.profit)/len(data.profit)
data["pro_rate"] = ["high" if i > threshold else "low" for i in data.profit]
#data.loc[:10,["speed_level","Speed"]]
data.head(50)
data.boxplot(column='profit',by = 'pro_rate')
plt.show()
data.shape
data2=data.head(10)
data2.head()
melted = pd.melt(frame=data2,id_vars = 'state', value_vars= ['profit','total_cost'])
melted
data["state"].value_counts(dropna =False)
data1=data.loc[:,["rd_spend","administration","marketing_spend"]]
data1.plot()
plt.show()
data1.plot(subplots=True)
plt.show()
data.plot(kind = "hist",y = "rd_spend",bins = 50,range= (0,200000),normed = True)
plt.show()
print(data.index.name)
data.index.name = "#"
data.head()
data1 = data.set_index(["pro_rate","profit"]) 
data1.head(100)
data1.groupby("state").mean()









