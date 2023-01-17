# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

% matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

# This is the 5 day Challenge with Rachael-Kaggle

# I will be using the CPU-GPU Evolution over time



#gpu_df = pd.read_csv("../input/All_GPUs.csv")

cpu_df = pd.read_csv("../input/Intel_CPUs.csv")

#gpu_df.head()

cpu_df.head()

cpu_df.describe()

cpu_df.shape

cpu_df.columns

cpu_df.shape

cpu_df.isnull().sum()
# There are lot off columns with null values so lets create a new df with intersting columns

newcpu = cpu_df.filter([ 'Vertical_Segment',  

                        'Status','Launch_Date', 'Lithography', 'nb_of_Cores', 'Processor_Base_Frequency',

                        'Cache', 'Bus_Speed', 'TDP', 

                       'Embedded_Options_Available'], axis=1)

newcpu ["Lithography"] = newcpu["Lithography"].str[:2]

newcpu["Lithography"].dtype

#mybins = (15,  30, 40, 80, 110, 300)

#mylabels = ("14nm", "22nm", "32nm","100nm", "200nm" )

#litho = newcpu["Lithography"]

#plt.hist (litho, mybins, labels=mylabels)

#plt.show()

#newcpu["BaseFreq"] = newcpu(["Processor_Base_Frequency"].str[5:] == "MHz").dropna()

#newcpu["BaseFreq"] = newcpu["Processor_Base_Frequency"].str[:4]

newcpu.head(10)



# Now lets see if the GPU or CPU dfs have any missing values

newcpu.isnull().any().any()

newcpu
newcpu.info
#Lets add columns and quarters columns

newcpu.Launch_Date = newcpu.Launch_Date.str.replace("\'", "-")

newcpu[["Quarter", "Year"]] = newcpu.Launch_Date.str.split("-", expand=True)

newcpu.dropna(subset=["Quarter"], inplace=True) 

# For some rows there were no Launch date: so we dropped those rows (about 400)

newcpu.head(5)
newcpu.dropna(subset = ["Lithography"],inplace=True)

newcpu.head(500)

litho_dictionary = {"14 nm": 14, '22 nm' : 22, "25 nm": 25, "28 nm": 28, "32 nm": 32,"45 nm":45,  "65 nm":65, "90 nm": 90, "100 nm":100, "130 nm": 130, "180 nm":180, "200 nm":200, "250 nm":250, "280 nm":280 }

newcpu["Lithography"] =newcpu["Lithography"].apply(lambda x: litho_dictionary[x])

newcpu.head()
# PLotting histograms and Bar charts 

bins = [15, 23, 26, 29, 33, 46, 66, 91, 110, 300]

mylabels = ["14nm", "22nm", "25nm", "28nm", "32nm", "45nm", "65nm", "90nm", "100nm"]

#plt.title ("Binning showing the Lithography counts")

newcpu.hist(column = ['Lithography'])

plt.show ()

Lithobins = pd.cut(newcpu.Lithography, bins, labels=mylabels)

fig, ax = plt.subplots(figsize=(11,7))

sns.countplot(x=Lithobins)

newcpu.head(10)

VS_dictionary = {"Mobile": 1, 'Desktop' : 2, "Embedded": 3, "Server":4 }

newcpu["VS"] = newcpu["Vertical_Segment"].apply(lambda x: VS_dictionary[x])

newcpu["Year"] = newcpu["Launch_Date"].str[-2:]

newcpu["Quarter"] = newcpu["Launch_Date"].str[:2]

newcpu.dropna(subset = ["Quarter"], inplace=True)

Q_dictionary ={"Q1":1, "Q2":2, "Q3":3, "Q4":4, "04":4}

newcpu["Q"] = newcpu["Quarter"].apply(lambda x: Q_dictionary[x])

newcpu.head(10)

#Now lets see which devices shiipped in which quarter using seaborn

sns.countplot(newcpu["Q"])

#idx = (newcpu.columns == 'Launch_Date').argmax()

#idx.dtype

#print (idx)

#newcpu.insert(idx, "Y", newcpu["Quarter"].apply(lambda x: Q_dictionary[x]))

#newcpu.head(100)

#group_q = newcpu["VS"].groupby(newcpu["Q"])

#group_q.head(10)

#ax = group_q[['Q', "VS"]].plot(kind='bar', title ="comp", figsize=(15, 10), legend=True, fontsize=12)

#ax.set_xlabel("Quarter", fontsize=12)

#ax.set_ylabel("VS", fontsize=12)

#plt.show()



#b=sns.barplot(x="Launch_Date",y="Vertical_Segment", data=newcpu)

#plt.show()
from scipy.stats import ttest_ind  # this is for ttest

from  scipy.stats import probplot # this is for qqplot



#probplot(cpu_df["Lithography"], dist="norm", plot=pylab)

#plt.show()