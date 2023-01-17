# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import csv

with open('../input/amazon_jobs_dataset.csv') as file_obj:

    file_data=csv.DictReader(file_obj, skipinitialspace=True)

    

    dic={}

    dic['IN, KA, Bangalore']=0

    dic['US, WA, Seattle ']=0

    for row in file_data:

        if row['location']=='IN, KA, Bangalore ':

            dic['IN, KA, Bangalore']+=1

        elif row['location']=='US, WA, Seattle ':

            dic['US, WA, Seattle ']+=1

    for i in dic:

        print(dic[i], end=' ')
with open('../input/amazon_jobs_dataset.csv', encoding='utf8') as file_obj:

    file_data=csv.DictReader(file_obj, skipinitialspace=True)

    lst=list(file_data)

    count=0

    for row in lst:

        a=row['Title'].split()

        if 'Vision' in a:

            count+=1

print(count)
with open('../input/amazon_jobs_dataset.csv', encoding='utf8') as file_obj:

    file_data=csv.DictReader(file_obj, skipinitialspace=True)

    lst=list(file_data)

    count=0

    for row in lst:

        a=row['location'].split()

        if not('US,' in a and 'CA,' in a) and 'CA,' in  a:

            count+=1

print(count)
with open('../input/amazon_jobs_dataset.csv', encoding='utf8') as file_obj:

    file_data=csv.DictReader(file_obj, skipinitialspace=True)

    lst=list(file_data)

    l=[]

    for row in lst:

        a=row['Posting_date'].split()

        if '2018' in a:

            l.append(a)

dic={}

for i in l:

    if i[0] in dic.keys():

        dic[i[0]]+=1

    else:

        dic[i[0]]=1

freq=0

maxmonth='January'

for i in dic:

    if dic[i]>freq:

        freq=dic[i]

        maxmonth=i

print(maxmonth, freq)
with open('../input/amazon_jobs_dataset.csv', encoding='utf8') as file_obj:

    file_data=csv.DictReader(file_obj, skipinitialspace=True)

    lst=list(file_data)

count=0

for row in lst:

    a=row['BASIC QUALIFICATIONS']

    if ("Bachelor" in a or "BA" in a or "BS" in a):

        count+=1

print(count)
with open('../input/amazon_jobs_dataset.csv', encoding='utf8') as file_obj:

    file_data=csv.DictReader(file_obj, skipinitialspace=True)

    lst=list(file_data)

count=0

dic={}

dic['Java']=0

dic['C++']=0

dic['Python']=0

for row in lst:

    loc=row['location'].strip().split(",")

    a=row['BASIC QUALIFICATIONS']

    if 'Java' in a and ("Bachelor" in a or "BA" in a or "BS" in a) and (loc[0]=='IN'):

        dic['Java']+=1

    elif 'C++' in a and ("Bachelor" in a or "BA" in a or "BS" in a) and (loc[0]=='IN'):

        dic['C++']+=1

    elif 'Python' in a and ("Bachelor" in a or "BA" in a or "BS" in a) and (loc[0]=='IN'):

        dic['Python']+=1

count=0

language=""

for i in dic.keys():

    if count<dic[i]:

        count=dic[i]

        language=i

print(language, count)
with open('../input/amazon_jobs_dataset.csv', encoding='utf8') as file_obj:

    file_data=csv.DictReader(file_obj, skipinitialspace=True)

    lst=list(file_data)

    dic={}

    for row in lst:

        a=row['BASIC QUALIFICATIONS']

        b=row['location'].split(",")[0]

        if "Java" in a:

            if b in dic.keys():

                dic[b]+=1

            else:

                dic[b]=1

                

count=0

country=""

for i in dic:

    if dic[i]>count:

        count=dic[i]

        country=i

print(country, count)
#importing important libraries

import numpy as np

import matplotlib.pyplot as plt

with open('../input/amazon_jobs_dataset.csv', encoding='utf8') as file_obj:

    file_data=csv.DictReader(file_obj, skipinitialspace=True)

    year=[]

    for row in file_data:

        year.append(row['Posting_date'].split()[2])

    np_year=np.array(year, dtype='int')

    dic=dict()

    for i in np_year:

        if i in dic.keys():

            dic[i]+=1

        else:

            dic[i]=1

    xaxis=[]

    yaxis=[]

    for i in dic.keys():

        xaxis.append(i)

        yaxis.append(dic[i])

    plt.plot(xaxis, yaxis, color='blue')

    plt.show()

    for i in range(len(xaxis)):

        print(xaxis[len(xaxis)-i-1], yaxis[len(xaxis)-i-1])

        

    
with open('../input/amazon_jobs_dataset.csv', encoding='utf8') as file_obj:

    file_data=csv.DictReader(file_obj, skipinitialspace=True)

    month=[]

    for row in file_data:

        month.append(row['Posting_date'].split()[0])

    np_month=np.array(month)

    dic=dict()

    for i in np_month:

        if i in dic:

            dic[i]+=1

        else:

            dic[i]=1

    xaxis=[]

    yaxis=[]

    for i in dic.keys():

        xaxis.append(i)

        yaxis.append(dic[i])

    

    plt.bar(xaxis, yaxis, color='orange')

    plt.xticks(rotation=45)

    plt.show()

    for i in range(len(xaxis)):

        print(xaxis[i], yaxis[i])
with open('../input/amazon_jobs_dataset.csv', encoding='utf8') as file_obj:

    file_data=csv.DictReader(file_obj, skipinitialspace=True)

    dic=dict()

    city=[]

    for row in file_data:

        if 'IN' in row['location'] and not 'IE' in row['location']:

            city.append(row['location'].split()[2])

    for i in city:

        if i in dic:

            dic[i]+=1

        else:

            dic[i]=1

            

    xaxis=[]

    yaxis=[]

    for i in dic:

        xaxis.append(i)

        yaxis.append(dic[i])

    np_xaxis=np.array(xaxis)

    np_yaxis=np.array(yaxis, dtype='int')

    

    np_xaxis=np_xaxis[np.argsort(np_yaxis)]

    np_yaxis=np.sort(np_yaxis)

    

    np_xaxis=np_xaxis[::-1]

    np_yaxis=np_yaxis[::-1]

    

    plt.pie(np_yaxis, labels=np_xaxis, autopct='%.2f%%', radius=2, explode=[0.1, 0.1, 0.1, 0.1, 0.8])

    plt.show()

    

    for i in range(len(np_xaxis)):

        print(np_xaxis[i], format((np_yaxis[i]*100)/sum(dic.values()), '.2f'))
with open('../input/amazon_jobs_dataset.csv', encoding='utf8') as file_obj:

    file_data=csv.DictReader(file_obj, skipinitialspace=True)

    year=[]

    for row in file_data:

        if 'java' in row['BASIC QUALIFICATIONS'] or 'Java' in row['BASIC QUALIFICATIONS']:

            year.append(row['Posting_date'].split()[2])

    np_year=np.array(year, dtype='int')

    dic=dict()

    for i in np_year:

        if i in dic.keys():

            dic[i]+=1

        else:

            dic[i]=1

    xaxis=list(dic.keys())

    yaxis=list(dic.values())

    

    np_xaxis=np.array(xaxis)

    np_yaxis=np.array(yaxis)

    

    np_xaxis=np_xaxis[::-1]

    np_yaxis=np_yaxis[::-1]

    

    plt.scatter(np_xaxis, np_yaxis)

    plt.show()

    

    for i in range(len(np_xaxis)):

        print(np_xaxis[i], np_yaxis[i])