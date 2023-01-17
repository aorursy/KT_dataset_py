# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from statsmodels.formula.api import ols

from glob import glob



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Q1

#Read data from multiple files, and merge necessary columns into a single dataframe.

def get_single_dataframe():

    fileNames = glob("../input/*.csv")

    dfs=[]

    for i in fileNames:

        df = pd.read_csv(i)

        df.columns = [c.lower().replace(' ','_') for c in df.columns]

        #get avarage price from High,Low,Open,Close price 

        df['price'] = (df.high + df.low + df.open + df.close) /4

        #add Product name column

        df['name'] = i.replace('.csv','').split('/')[2]

        #delete unnecessary columns

        df.drop(['high','low','open','close'], axis=1,inplace=True)

        dfs.append(df)

    dfs = pd.concat(dfs)

    return dfs



singleDataFrame = get_single_dataframe()

singleDataFrame.groupby('name').describe()

#singleDataFrame.hist()

#Q2

#Create and print simple correlation matrix to identify perfect substitute and complementtary products.



print(singleDataFrame.corr())

p_list = singleDataFrame.name.unique()

ndfs = []

for p1 in p_list:

    p1f = singleDataFrame[singleDataFrame['name'] == p1]

    for p2 in p_list:

        if(p1 == p2):

            continue

        p2f = singleDataFrame[singleDataFrame['name'] == p2]

        ndfs.append([p1 + ' ' + p2, p1f.volume.corr(p2f.price)])

        #print(p1 + ' ' + p2) 

        #print(pd.concat([p1f['volume'], p2f['price']], axis=1).corr().iloc[1:2,:1])

ndfs = pd.DataFrame(ndfs,columns=['products','corr'])

ndfs
#Q3

#Conduct linear regression to analyse imperfect substitute and complementary products.



for p1 in p_list:

    p1f = singleDataFrame[singleDataFrame['name'] == p1]

    for p2 in p_list:

        if p1 == p2:

            continue

        p2f = singleDataFrame[singleDataFrame['name'] == p2]

        vol = pd.concat([p1f["volume"], p2f["price"]],axis=1)

        ols_analysis = ols("volume ~ price", data=vol).fit()

        print(ols_analysis.summary())

        






    