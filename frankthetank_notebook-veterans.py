# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

filenames = check_output(["ls", "../input"]).decode("utf8")

filenames = filenames.split("\n")

# Any results you write to the current directory are saved as output.
#Results is dataframe with all the files put together with a Year column

list_=[]

alldata=pd.DataFrame()

for file in filenames[0:-2]:

    year = file.split('.')[0]

    filepath="../input/"+file.strip()

    mydata = pd.read_csv(filepath)

    mydata["Year"]=year

    list_.append(mydata)

    print(year, "US veterans: ",sum(mydata.vet_pop), ", Vet suicides:", sum(mydata.vet_suicides))

alldata=pd.concat(list_)

alldata=alldata.fillna('0')

print(alldata.head())

print(alldata.describe())



# civ_rate, vet_rate - meaning x per 100,000 will commit suicide


alldata[['Year','state', 'vet_suicides']]



sns.set(style="ticks")

# Load the example tips dataset

# tips = sns.load_dataset("tips")



sns.boxplot(x="state", y="vet_suicides", data=alldata, palette="PRGn")

sns.despine(offset=10, trim=True)



#sns.boxplot(x="state", y="vet_suicides_p", data=alldata, palette="PRGn")

#sns.despine(offset=10, trim=True)