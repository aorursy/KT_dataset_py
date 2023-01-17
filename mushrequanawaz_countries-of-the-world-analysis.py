# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/countries-of-the-world/countries of the world.csv")
data
data.info()
#solving comma issues
#code
def replace_(l):
    newl=[]
    for i in l:
        if(str(i).lower!='nan'):
            newl.append(float(str(i).replace(',','.')))
        else:
            newl.append(np.nan)
    return newl
        
#implementation
data['Pop. Density (per sq. mi.)']= replace_(data['Pop. Density (per sq. mi.)'])
data['Coastline (coast/area ratio)']= replace_(data['Coastline (coast/area ratio)'])
data['Net migration']=replace_(data['Net migration'])
data['Infant mortality (per 1000 births)']=replace_(data['Infant mortality (per 1000 births)'])
data["Literacy (%)"] = replace_(data["Literacy (%)"])
data["Arable (%)"] = replace_(data["Arable (%)"])
data["Crops (%)"] = replace_(data["Crops (%)"])
data["Other (%)"] = replace_(data["Other (%)"])
data["Birthrate"] = replace_(data["Birthrate"])
data["Deathrate"] = replace_(data["Deathrate"])
data["Agriculture"] = replace_(data["Agriculture"])
data["Industry"] = replace_(data["Industry"])
data["Service"] = replace_(data["Service"])
data["Phones (per 1000)"] = replace_(data["Phones (per 1000)"])
data["Climate"] = replace_(data["Climate"])

#datatype issues
data["Region"] = data["Region"].astype("category")
data["Climate"] = data["Climate"].astype("category")
data["Area (sq. mi.)"] = data["Area (sq. mi.)"].astype("float")
#test
data
data.info()
data.sample()
data["Net migration"].fillna(data["Net migration"].mean(), inplace=True)
data["Infant mortality (per 1000 births)"].fillna(data["Infant mortality (per 1000 births)"].mean(), inplace=True)
data["GDP ($ per capita)"].fillna(data["GDP ($ per capita)"].mean(), inplace=True)
data["Literacy (%)"].fillna(data["Literacy (%)"].mean(), inplace=True)
data["Phones (per 1000)"].fillna(data["Phones (per 1000)"].mean(), inplace=True)
data["Arable (%)"].fillna(data["Arable (%)"].mean(), inplace=True)
data["Crops (%)"].fillna(data["Crops (%)"].mean(), inplace=True)
data["Other (%)"].fillna(data["Other (%)"].mean(), inplace=True)
data["Birthrate"].fillna(data["Birthrate"].mean(), inplace=True)
data["Deathrate"].fillna(data["Deathrate"].mean(), inplace=True)
data["Agriculture"].fillna(data["Agriculture"].mean(), inplace=True)
data["Industry"].fillna(data["Industry"].mean(), inplace=True)
data["Service"].fillna(data["Service"].mean(), inplace=True)
data["Climate"].value_counts()
data["Climate"].fillna(2.0, inplace=True)

data.describe()
data
#for discreet data
import matplotlib.pyplot as plt
import seaborn as sns
def assess_discreet(x):
    print(x.value_counts())
    plt.title('boxplot')
    sns.boxplot(x)
    plt.show()
    
    x.value_counts().plot(kind='pie', autopct='%0.2f')
    plt.show()
    total = data.shape[0]

    print((x.value_counts()/total)*100)
    sns.countplot(x)
    
assess_discreet(data['Climate'])
#for continuous data
def assess_continuous(x):
    sns.violinplot(x,color='red')
    plt.title('Violin plot ')
    plt.show()
    sns.distplot(x,rug=True,color='green')
    plt.title('Distplot')
    plt.show()
    plt.title('boxplot')
    sns.boxplot(x)
    plt.show()
    

assess_continuous(data['Area (sq. mi.)'])
assess_continuous(data['Pop. Density (per sq. mi.)'])
assess_continuous(data['Coastline (coast/area ratio)'])
assess_continuous(data['Net migration'])
assess_continuous(data['Infant mortality (per 1000 births)'])
assess_continuous(data['GDP ($ per capita)'])
assess_continuous(data['Literacy (%)'])
assess_continuous(data['Phones (per 1000)'])
assess_continuous(data['Arable (%)'])
assess_continuous(data['Crops (%)'])
assess_continuous(data['Other (%)'])
assess_continuous(data['Birthrate'])
assess_continuous(data['Deathrate'])
assess_continuous(data['Agriculture'])
assess_continuous(data['Industry'])
assess_continuous(data['Service'])
columns = data.columns[2:]
for i in columns:
    if(data[i].dtype.name == "category"):
        print(i)
        sns.countplot(data['Net migration'], hue=data[i])
        
data
for i in columns:
    if(data[i].dtype.name != "category"):
        print(i)
        sns.relplot(x=i, y="GDP ($ per capita)", data=data)
        
    plt.show()

sns.heatmap(data.corr())
