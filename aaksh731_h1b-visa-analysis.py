# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import re

# Any results you write to the current directory are saved as output.
df=pd.read_csv('..//input//h1b_kaggle.csv')
df.head(1)
def get_normalized_string(x):

    try:

        string=re.sub(r"[^a-zA-Z0-9]+", ' ', x)

        return(string)

    except:

        return('')
df['CASE_STATUS']=df['CASE_STATUS'].map(lambda x: get_normalized_string(x))

df['EMPLOYER_NAME']=df['EMPLOYER_NAME'].map(lambda x: get_normalized_string(x))

df['JOB_TITLE']=df['JOB_TITLE'].map(lambda x: get_normalized_string(x))

df['WORKSITE']=df['WORKSITE'].map(lambda x: get_normalized_string(x))



df['SOC_NAME']=df['SOC_NAME'].map(lambda x: get_normalized_string(x))

df.head()
#Is the number of petitions with Data Engineer job title increasing over time?

plt.plot(df[df['JOB_TITLE']=='DATA ENGINEER'][['JOB_TITLE','YEAR']].groupby(by=['YEAR']).count())

#Answer -> Yes
#Which part of the US has the most Hardware Engineer jobs? 

countlist=[]

for i ,j in df[df['JOB_TITLE']=='HARDWARE ENGINEER'][['JOB_TITLE','WORKSITE']].groupby(by=['WORKSITE']):

    countlist.append([i,len(j)])

countdf=pd.DataFrame(countlist,columns=['WORKSITE','count'])

#countdf=countdf.set_index('WORKSITE')                                           

countdf=countdf.sort_values(by=['count'],ascending=False)

sns.barplot(data=countdf.head(10),y='WORKSITE',x='count',orient='h')

#Which industry has the most number of Data Scientist positions?

countlist=[]

df['SOC_NAME']=df['SOC_NAME'].map(lambda x:str(x).title())

for i ,j in df[df['JOB_TITLE']=='DATA SCIENTIST'][['SOC_NAME','JOB_TITLE']].groupby(by=['SOC_NAME']):

    countlist.append([i,len(j)])

countdf=pd.DataFrame(countlist,columns=['SOC_NAME','count'])

#countdf=countdf.set_index('WORKSITE')                                           

countdf=countdf.sort_values(by=['count'],ascending=False)

sns.barplot(data=countdf.head(10),y='SOC_NAME',x='count',orient='h')



plt.show()

#Which employers file the most petitions each year?

countlist=[]

company_total_df=[]

for i,j in df.groupby(by=['EMPLOYER_NAME']):

    company_total_df.append([i,len(j)])

    for k,q in j.groupby(by=['YEAR']):

        countlist.append([i,k,len(q)])



countlistdf=pd.DataFrame(countlist,columns=['Emp_name','year','count'])

company_total_df_dataframe=pd.DataFrame(company_total_df,columns=['Emp_name','count'])



                                           

company_total_df_dataframe=company_total_df_dataframe.sort_values(by=['count'],ascending=False)

names=company_total_df_dataframe.head(10)['Emp_name'].tolist()

top10_comp=countlistdf[countlistdf['Emp_name'].isin(names)]

hash_index=[]

for i in names:

    hash_index+=list(top10_comp[top10_comp['Emp_name']==i].index)

top10_comp=top10_comp.loc[hash_index]

sns.barplot(data=top10_comp,x='Emp_name',y='count',hue='year',orient='v')

plt.xticks(rotation=90)



plt.show()

#Different category top 10 company

countlist=[]

for i,j in df.groupby(by=['EMPLOYER_NAME']):

    for k,q in j.groupby(by=['YEAR']):

        total=len(q)

        for a,b in q.groupby(by=['CASE_STATUS']):

            countlist.append([i,k,a,len(b)/total])

countlistdf=pd.DataFrame(countlist,columns=['Emp_name','year','status','count'])

top_50=company_total_df_dataframe.head(50)['Emp_name'].tolist()

for i in sorted(countlistdf['year'].unique().tolist()):

    for j in sorted(countlistdf.status.unique().tolist()):

        

       

        try:

            dummy=pd.pivot_table(countlistdf[(countlistdf['Emp_name'].isin(top_50))&

                                  (countlistdf['year']==i)],

                                 index=['Emp_name','year'], 

                       columns='status', values='count').sort_values(by=[j],ascending=False)[[j]].head(10)

            sns.barplot(data=dummy,y=dummy.index.get_level_values('Emp_name'),x=j,orient='h')

            

            plt.xticks(rotation=90)

            plt.title('{} {}'.format(j,i))

            plt.figure()

        except:

            pass

plt.show()