# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import random
ItemName={"Item 1":"Ideal L2 self (7 questions)",

         "Item 2":"Ought-To L2 self (8 ques)",

         "Item 3":"Parental encouragement (5 ques)",

         "Item 4":"Instrumental-promotion (6 ques)",

        "Item 5":"Instrumentality-prevention (5 ques)",

         "Item 6":"Attitudes towards learning English (7 ques)",

        "Item 7":"Travel orientation (3 ques)",

        "Item 8":"Fear of assimilation(5 ques)",

        "Item 9":"Ethnocentrism (5 ques)",

        "Item 10":"English anxiety (4 ques)",

        "Item 11":"Integrativeness (3 ques)",

        "Item 12":"Cultural interest (4 ques)",

        "Item 13":"Attitudes toward L2 community (3 ques)"}



ItemDict={"Item 1":[6, 14, 23, 29, 37, 38, 46],

"Item 2":[5, 12, 19, 27,35, 36, 43, 48, 49],

"Item 3":[2, 11, 21, 30, 40],

"Item 4":[4, 10, 16, 22, 28, 41],

"Item 5":[7, 18, 25, 33, 42],

"Item 6":[3, 13, 31, 45, 50, 55, 60, 65],

"Item 7":[20, 47, 64],

"Item 8":[8, 17, 26, 32, 34 ],

"Item 9":[9, 15, 24, 39, 44],

"Item 10":[51, 56, 61, 66],

"Item 11":[52, 57, 62,],

"Item 12":[53, 54, 57, 58],

"Item 13":[1, 59, 67]}
data_file = "../input/data621.csv"

Data=pd.read_csv(data_file,low_memory=False)
#  caculate item

df1=Data[['序号', 'Your Gender:', 'Your Age', 'Do you live in urban or rural are?','Have you ever taken IELTS or TOFEL tests?']]

import copy

for item in ItemDict:

    s=copy.deepcopy(Data.iloc[:,ItemDict[item][0]+5])

    for k in ItemDict[item][1:]:

        s+=Data.iloc[:,k+5]

    s = s/len(ItemDict[item])

    df1[item]=s

    

MaleData=df1[df1["Your Gender:"]=="Male"]

FemaleData=df1[df1["Your Gender:"]=="Female"]

Items=list(df1.columns)[5:]

print(Items)
from scipy.stats import ttest_ind

import matplotlib.pyplot as plt

import seaborn as sns

f,axs=plt.subplots(nrows=3,ncols=4,figsize=(20, 15))



for item,ax in zip(["Item %s"%i for i in range(1,13)],axs.flat):

    s,p=ttest_ind(MaleData[item],FemaleData[item])

    star=""

    m1,m2=MaleData[item].mean(),FemaleData[item].mean()

    for i in range(1,4):

        if p<=10**(-i):

            star="*"*i

    print("%s T-test: statistic=%f,pvalue=%f%s  Male's mean : %f Female's mean: %f"%(item,s,p,star,m1,m2))

    sns.boxplot(x="Your Gender:", y=item, data=df1,ax=ax)

    ax.set_xlabel(" ",fontsize=20)

    ax.set_ylabel("Score",fontsize=20)

    ax.set_title(item,fontsize=20)

    ax.set_ylim([0,7])

plt.show()
item= "Item 13"

s,p=ttest_ind(MaleData[item],FemaleData[item])

star=""

for i in range(1,4):

    if p<=10**(-i):

        star="*"*i

plt.figure(figsize=(5,5))

m1,m2=MaleData[item].mean(),FemaleData[item].mean()

print("%s T-test: statistic=%f,pvalue=%f%s Male's mean : %f Female's mean: %f"%(item,s,p,star,m1,m2))

sns.boxplot(x="Your Gender:", y=item, data=df1)  

plt.xlabel(" ",fontsize=20)

plt.ylabel("Score",fontsize=20)

plt.title(item.split(":")[0],fontsize=20)

plt.ylim([0,7])

plt.show()