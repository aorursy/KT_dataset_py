# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
mcr = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')

qs = pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')
print(mcr.shape, qs.shape)
mcr.head()
for i in qs.columns:

    print(i,' : ',qs[i][0])
mcr_indian = mcr[mcr['Q3']=='India']
mcr_indian.shape
mcr_indian.shape[0]/mcr.shape[0]
mcr_indian.head()
#Reset_index

mcr_indian = mcr_indian.reset_index().drop('index', axis=1)
mcr_indian.head()
for i in mcr_indian.isna().sum().index:

    print(i,' : ', mcr_indian[i].isna().sum())
def question(i):

    if i<=34:

        print('Question ',i, ' : ')

        return qs[qs.columns[i]][0]

    else:

        print('No such question exist')
question(1)
question(0)
duration_only = mcr_indian[mcr_indian.columns[0]]
duration_only.isna().sum()
max(pd.to_numeric(duration_only)/60)
import matplotlib

matplotlib.rcParams.update({'font.size': 24})
import seaborn as sns

lst = pd.to_numeric(duration_only)/60

lst = pd.DataFrame(lst)

lst.columns = ['Time from Start to Finish (minutes)']

lst = lst['Time from Start to Finish (minutes)']

sns.distplot(lst, bins = 5000).set(xlim=(0, 60))
print('There are {} total indian respondents out of whom {} respondents have spent more than 10 minutes taking this survey and {} respondents really took more than an hour.'.format(lst.shape[0],lst[lst>10].shape[0],lst[lst>60].shape[0]))


def plotMcqHist(i):

    print(mcr_indian.columns[i], '  : ', mcr[mcr_indian.columns[i]][0])

    temp = mcr_indian[mcr_indian.columns[i]].value_counts()

    temp = temp.fillna(-10)

    plt.figure(figsize=(20,5))

    plt.xlabel(mcr_indian.columns[i])

    plt.bar(list(temp.index),list(temp))

    

    plt.xticks(rotation=90)

    plt.show()

    print('='*50)

    
mcr_indian.head()
for i in enumerate(mcr_indian.columns):

    print(i)
lst1 = [1,2,5,6,8,9,10,20,21,48,55,95,116,117]
plotMcqHist(lst1[0])
print("There are {} % of Indian respondents who belong to 18 to 21 Age group".format(round((((mcr_indian[mcr_indian.columns[lst1[0]]]=='18-21')).sum()/len(mcr_indian))*100,2)))

print("There are {} % of Indian respondents who belong to 22 to 24 Age group".format(round((((mcr_indian[mcr_indian.columns[lst1[0]]]=='22-24')).sum()/len(mcr_indian))*100,2)))

print("There are {} % of Indian respondents who belong to 25 to 29 Age group".format(round((((mcr_indian[mcr_indian.columns[lst1[0]]]=='25-29')).sum()/len(mcr_indian))*100,2)))

plotMcqHist(lst1[1])
print("{} % of Indian respondents are Male".format(round((((mcr_indian[mcr_indian.columns[lst1[1]]]=='Male')).sum()/len(mcr_indian))*100,2)))

print("{} % of Indian respondents are Female".format(round((((mcr_indian[mcr_indian.columns[lst1[1]]]=='Female')).sum()/len(mcr_indian))*100,2)))
print("There are {} % of Indian respondents who belong to 18 to 21 Age group and are Female".format(round((((mcr_indian[mcr_indian.columns[lst1[0]]]=='18-21') & (mcr_indian[mcr_indian.columns[lst1[1]]]=='Female')).sum()/len(mcr_indian))*100,2)))
print("There are {}% of Indian respondents who belong to 18 to 29 Age group and are Male".format(round(((((mcr_indian[mcr_indian.columns[lst1[0]]]=='18-21') | (mcr_indian[mcr_indian.columns[lst1[0]]]=='22-24') | (mcr_indian[mcr_indian.columns[lst1[0]]]=='25-29')) & (mcr_indian[mcr_indian.columns[lst1[1]]]=='Male')).sum()/len(mcr_indian))*100,2)))

print("There are {}% of Indian respondents who belong to 18 to 29 Age group and are Female".format(round(((((mcr_indian[mcr_indian.columns[lst1[0]]]=='18-21') | (mcr_indian[mcr_indian.columns[lst1[0]]]=='22-24') | (mcr_indian[mcr_indian.columns[lst1[0]]]=='25-29')) & (mcr_indian[mcr_indian.columns[lst1[1]]]=='Female')).sum()/len(mcr_indian))*100,2)))
plotMcqHist(lst1[2])
print("{} % of Indian respondents have Bachelors as Highest level of Formal education".format(round((((mcr_indian[mcr_indian.columns[lst1[2]]]=='Bachelor’s degree')).sum()/len(mcr_indian))*100,2)))

print("{} % of Indian respondents have Masters as Highest level of Formal education".format(round((((mcr_indian[mcr_indian.columns[lst1[2]]]=='Master’s degree')).sum()/len(mcr_indian))*100,2)))

print("{} % of Indian respondents have Masters as Highest level of Formal education".format(round((((mcr_indian[mcr_indian.columns[lst1[2]]]=='Doctoral degree')).sum()/len(mcr_indian))*100,2)))
plotMcqHist(lst1[3])
print("{} % of Indian respondents are students".format(round((((mcr_indian[mcr_indian.columns[lst1[3]]]=='Student')).sum()/len(mcr_indian))*100,2)))

print("While {} % of Indian respondents are not employed".format(round((((mcr_indian[mcr_indian.columns[lst1[3]]]=='Not employed')).sum()/len(mcr_indian))*100,2)))
plotMcqHist(lst1[4])
print("{} % of Indian respondents are employed and belong to a company size <=49 employees".format(round((((mcr_indian[mcr_indian.columns[lst1[4]]]=='0-49 employees')).sum()/len(mcr_indian))*100,2)))

print("{} % of Indian respondents are employed and belong to a company size >10,000 employees".format(round((((mcr_indian[mcr_indian.columns[lst1[4]]]=='> 10,000 employees')).sum()/len(mcr_indian))*100,2)))

print("{} % of Indian respondents are employed".format(round((((mcr_indian[mcr_indian.columns[lst1[4]]]=='0-49 employees') | (mcr_indian[mcr_indian.columns[lst1[4]]]=='> 50-249 employees') | (mcr_indian[mcr_indian.columns[lst1[4]]]=='250-999 employees') | (mcr_indian[mcr_indian.columns[lst1[4]]]=='1000-9,999 employees') | (mcr_indian[mcr_indian.columns[lst1[4]]]=='> 10,000 employees')).sum()/len(mcr_indian))*100,2)))
plotMcqHist(lst1[5])
plotMcqHist(lst1[6])