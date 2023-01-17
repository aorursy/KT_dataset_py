# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sbn

pd.set_option('display.max_columns', 5000)

pd.set_option('display.max_rows',5000)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
family_df = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv')

sample_sub = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/sample_submission.csv')

sample_sub.head()
family_df.head()
family_df.n_people.value_counts().plot(kind = 'bar',figsize = (15,7))
family_df.isnull().sum()
family_df[family_df['choice_0']> 100]
sbn.pairplot(family_df)
sbn.distplot(family_df.choice_0)
plt.figure(figsize = (16,5))

sbn.distplot(family_df.choice_1)
plt.figure(figsize=(16, 6))

ax = sbn.distplot(family_df.choice_1)

plt.figure(figsize = (16,6))

sbn.boxplot(x = family_df.n_people,y = family_df.choice_0)
plt.figure(figsize = (16,6))

sbn.boxplot(x = family_df.n_people,y = family_df.choice_9)
plt.figure(figsize = (16,6))

sbn.boxplot(x = family_df.n_people,y = family_df.choice_2)
family_df[family_df.choice_0==1]['n_people'].sum()
family_df.groupby(['choice_0'])['n_people'].agg(sum).plot(kind = 'bar',figsize = (16,6))


for index, row in family_df.iterrows():

    #print(row)

    hash_value = {}

    for (columnName, columnData) in row.iteritems():

        if (columnName != 'family_id') & (columnName != 'n_people'):

            hash_value[columnData] = 1+ hash_value.get(columnData,0)

            if hash_value[columnData] > 1:

                row[columnName] = -1

                #print(columnName)



            

 
family_df.values[(family_df == -1).values]
family_df.groupby(['choice_1'])['n_people'].agg(sum).plot(kind = 'bar',figsize = (16,6))
family_df.groupby(['choice_2'])['n_people'].agg(sum).plot(kind = 'bar',figsize = (16,6))
days_people = {}

for index, row in family_df.iterrows():

    for (columnName, columnData) in row.iteritems():

        if (columnName != 'family_id') & (columnName != 'n_people'):

            days_people["days_"+str(columnData )] = row['n_people'] + days_people.get("days_"+str(columnData),0)





               
days_people_df = pd.DataFrame.from_dict(list(days_people.items()))

days_people_df.columns = ['days_before_xmas','n_people_interested']
days_people_df.sort_values(by=['n_people_interested'], ascending=False).plot(x = 'days_before_xmas',y ='n_people_interested',  kind = 'bar',figsize = (16,6),)
(family_df.n_people.sum(),family_df.shape)
days_family = {}

for index, row in family_df.iterrows():

    for (columnName, columnData) in row.iteritems():

        if (columnName != 'family_id') & (columnName != 'n_people'):

            days_family["days_"+str(columnData )] = 1 + days_family.get("days_"+str(columnData),0)

            

days_family_df = pd.DataFrame.from_dict(list(days_family.items()))

days_family_df.columns = ['days_before_xmas','n_family_interested']
days_family_df.sort_values(by=['n_family_interested'], ascending=False).plot(x = 'days_before_xmas',y ='n_family_interested',  kind = 'bar',figsize = (16,6),)
family_df.n_people.sum()
def check_lowest_cost_day(row,n_alloc):

    flag = 0

    min_people = 100000

    day = 101

    for columnName,columnData in row.iteritems():

        if (columnName != 'family_id') and (columnName != 'n_people'):

            if flag == 0:

                flag = 1

                min_ = n_alloc.get(columnData,0) + row['n_people'] 

            else:

                min_ = n_alloc.get(columnData,0) + row['n_people'] 

            if (min_people > min_) and (min_ < 300):

                min_people = min_

                day = columnData

    

    if min_people == 100000:

        #print("outer")

        return min(n_allocated, key=n_allocated.get)

    return day

                        

            
results_dict = {}

n_allocated = {i:0 for i in range(1,101)}



for index, row in family_df.iterrows():

    day = check_lowest_cost_day(row,n_allocated)

    results_dict[row['family_id']] = day

    n_allocated[day] = n_allocated.get(day,0) + row['n_people']

    
results = pd.DataFrame.from_dict(list(results_dict.items()))

results.columns = ['family_id','assigned_day']
results_famil_df = results.merge(family_df,on='family_id',how = 'inner')
results_famil_df.groupby(['assigned_day'])['n_people'].agg(sum)
results.assigned_day.value_counts().plot(kind = 'bar', figsize = (16,6))
results.to_csv('submission.csv', index=False)