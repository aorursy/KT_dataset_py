# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

import ast

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/audata.csv',usecols=['res','studentdetail'])
#Required to deal with the data, which is in a weird format.

df['res'] = df['res'].apply(lambda x : ast.literal_eval(x))

df['studentdetail'] = df['studentdetail'].apply(lambda x : ast.literal_eval(x))
len(df) #Including the people who have WHx in their
df=df[df['res'].apply(lambda x : bool(x))] #Drop the
len(df) #Actual results length
marks = { 'U' : 0, 'E' : 2, 'D' : 3, 'C' : 4, 'B' : 5, 'A' : 6, 'S' : 7, 'UA' : 0 } #Make a map for the average

def extract_average(x):

    val=0

    tempdata =  x['res']

    for key,value in tempdata.items():

        val=val+marks[value]

    return round(val/len(tempdata)) 

df['average'] = df.apply(extract_average,axis=1) #Get the average
df['average'].value_counts().plot(kind='bar')
with open('../input/clgdis.json','r') as fp:

    clgdis = json.load(fp)

#Load map of college and districts

df['collegecode'] = df['studentdetail'].apply(lambda x : x[0][:4]) #Extract the college code

df['area'] = df['collegecode'].apply(lambda x : clgdis[x])

df.groupby('area')['average'].agg([np.mean]).plot(kind='bar')

df['departmentcode'] = df['studentdetail'].apply(lambda x:x[0][6:9])

df.groupby('departmentcode')['average'].agg([np.mean]).plot(kind='bar')
##Find the number of people who don't write exams per subject

#Make a map of subjects

def mapfunc(x):

    for sub,val in x.items():

        if sub in exams:

            exams[sub] = exams[sub] + 1

        else:

            exams[sub] =1

                

                

exams=dict()

d=df['res'].apply(mapfunc)

del d

#Find the number of UA per subject

subuacount =dict()

for x in exams.keys():

    subuacount[x] = 0

def getuafunc(x):

    for sub,val in x.items():

        if val == 'UA':

            subuacount[sub] = subuacount[sub] + 1

d=df['res'].apply(getuafunc)

del d

subuacount_df = pd.DataFrame(list(subuacount.items()),columns=['sub','count'])

subuacount_df.sort('count',ascending=False).head(20).plot(kind='bar',x='sub',y='count')