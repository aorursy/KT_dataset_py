# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data15=pd.read_csv("/kaggle/input/world-happiness/2015.csv")
data16=pd.read_csv("/kaggle/input/world-happiness/2016.csv")
data18=pd.read_csv("/kaggle/input/world-happiness/2018.csv")
data17=pd.read_csv("/kaggle/input/world-happiness/2017.csv")
data19=pd.read_csv("/kaggle/input/world-happiness/2019.csv")
data15=data15.rename(columns = {'Happiness Score':'Happiness_Score'})
data15=data15.rename(columns = {'Economy (GDP per Capita)':'Economy'})
data19=data19.rename(columns= {'Country or region':'Country'})
data19=data19.rename(columns= {'Freedom to make life choices':'Freedom'})
data19=data19.rename(columns= {'Score':'Happiness_Score'})
data17=data17.rename(columns= {'Happiness.Score':'Happiness_Score'})
data15=data15.rename(columns = {'Health (Life Expectancy)':'Health'})
data15.info()
data15.columns
data15.head(7)
data15.corr()
#correlation map
f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(data15.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data15.plot(kind='scatter',x='Freedom',y='Happiness_Score',alpha=0.9,color="purple",grid=True)
plt.legend()
plt.xlabel('Özgürlük')
plt.ylabel('Mutluluk')
plt.title('Özgürlük ve mutluluk arasındaki ilişki')
plt.show()
#correlationla ilgilendiğimiz için scatter tercih ettim.
Turkey=data15[data15.Country=='Turkey']
Canada=data15[data15.Country=='Canada']
Germany=data15[data15.Country=='Germany']
Cambodia=data15[data15.Country=='Cambodia']
plt.scatter(Turkey.Freedom,Turkey.Happiness_Score,color="purple",label="Turkey")
plt.scatter(Canada.Freedom,Canada.Happiness_Score,color="black",label='Canada')
plt.scatter(Germany.Freedom,Germany.Happiness_Score,color="orange",label='Germany')
plt.scatter(Cambodia.Freedom,Cambodia.Happiness_Score,color="yellow",label='Cambodia')
plt.scatter(data15.Freedom,data15.Happiness_Score,alpha=0.2,color="blue",label="World")
plt.legend()
plt.xlabel('Özgürlük')
plt.ylabel('Mutluluk')
plt.title('Özgürlük ve mutluluk arasındaki ilişki')
plt.show()
plt.scatter(Turkey.Economy,Turkey.Health,color="purple",label="Turkey")
plt.scatter(Germany.Economy,Germany.Health,color="orange",label='Germany')
plt.scatter(data15.Economy,data15.Health,alpha=0.07,color="blue",label="World")
plt.legend()
plt.xlabel('Ekonomi')
plt.ylabel('Sağlık')
plt.title('Ekonomi ve Sağlık arasındaki ilişki')
plt.show()
plt.scatter(Turkey.Economy,Turkey.Happiness_Score,color="purple",label="Turkey")
plt.scatter(Germany.Economy,Germany.Happiness_Score,color="orange",label='Germany')
plt.scatter(data15.Economy,data15.Happiness_Score,alpha=0.09,color="blue",label="World")
plt.legend()
plt.xlabel('Ekonomi')
plt.ylabel('Mutluluk')
plt.title('Ekonomi ve mutluluk arasındaki ilişki')
plt.show()
Turkey=data15[data15.Country=='Turkey']
Turkey17=data17[data17.Country=='Turkey']
Turkey19=data19[data19.Country=='Turkey']
plt.scatter(Turkey.Freedom,Turkey.Happiness_Score,color="purple",label="Turkey15")
plt.scatter(Turkey17.Freedom,Turkey17.Happiness_Score,color="pink",label='Turkey17')
plt.scatter(Turkey19.Freedom,Turkey17.Happiness_Score,color="orange",label='Turkey19')
plt.scatter(data15.Freedom,data15.Happiness_Score,alpha=0.06,color="blue",label="World")
plt.legend()
plt.xlabel('Özgürlük')
plt.ylabel('Mutluluk')
plt.title('Özgürlük ve mutluluk arasındaki ilişki')
plt.show()
data15[(data15['Freedom']<0.1) & (data15['Freedom']>0.0)]
data15.Freedom.max()
data15[(data15['Freedom']<1) & (data15['Freedom']>0.65)]
data15.tail()
data15.describe()
Turkey.loc[:,:]
data15.boxplot(column="Happiness_Score", by="Region")
plt.xticks(rotation=90)#bu kodu kullanmadığımız taktirde isimler yatay duruyor ve içi içe gelerek okunamaz hale geliyordu.
data15[data15.Region=='North America']
data15.boxplot(column="Happiness_Score")
data15.Health.plot(kind='hist',bins=50,figsize=(8,8))
def plot_hist(a):
    plt.figure(figsize=(10,3))
    plt.hist(data15[a],bins=20)
    plt.xlabel(a)
    plt.title("{}".format(a))
    plt.show

veriler="Health","Freedom", "Economy", "Family"
for i in veriler:
    plot_hist(i)