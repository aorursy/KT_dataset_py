import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df=pd.read_csv("../input/monster_com-job_sample.csv")
df.info()
df=df.drop(['country','country_code','job_board','has_expired'], axis=1)
df=df.drop(['page_url','uniq_id'],axis=1)
df.describe()
l=df.location.unique()
unique_location={}
for i in range(len(l)):
    unique_location[i]=l[i]
    
    
for k in df.organization.keys():
        if  df.organization[k] in l:
            temp=df.organization[k]
            df.organization[k]=df.location[k]
            df.location[k]=temp

df=df[df['location'].apply(lambda x: len(x)<20)]
location=df['location'].str.split(',')
df['location']=location.str[0]
pattern = r'[A-Z/a-z]'
df=df[df['location'].str.contains(pattern)]           
jobtype=df['job_type'].str.split(',')
df['job_type']=jobtype.str[0]
df['job_type'][df['job_type']=='Full Time Employee']='Full Time'
df['job_type'][df['job_type']=='Part Time Employee']='Part Time'
def min_max_hourly_wage(val):
    if pd.isnull(val):
        return np.nan
    elif ' /hour' in val:
        i=val.split('/hour')[0].replace('$',' ').strip()
        if '-' in i:
            mn=i.split('-')[0]
            mx=i.split('-')[1]
#Since there are several commas in salary. It will take it as a string. So it will throw an 
#error saying "unsupported operand type(s) for /: 'str' and 'int'" and will not be able to 
#perform operation. And, I used try catch block because there is € currency in salary also. So in order 
#to deal with this I preferred try catch block.
            try:
                mn = float(mn.replace(",", "").strip())
                mx = float(mx.replace(",", "").strip())
            except:
                return np.nan
            return mn, mx


def min_max_yearly_wage(val):
    if pd.isnull(val):
        return np.nan
    elif ' /year' in val:
        i=val.split('/year')[0].replace('$',' ').strip()
        if '-' in i:
            mn=i.split('-')[0]
            mx=i.split('-')[1]
#Since there are several commas in salary. It will take it as a string. So it will throw an 
#error saying "unsupported operand type(s) for /: 'str' and 'int'" and will not be able to 
#perform operation. And, I used try catch block because there is € currency in salary also. So in order 
#to deal with this I preferred try catch block.
            try:
                mn = float(mn.replace(",", "").strip())
                mx = float(mx.replace(",", "").strip())
            except:
                return np.nan
            return mn, mx
        
        
df = df.assign(yearly_salary_range=df['salary'].apply(min_max_yearly_wage),
                   hourly_salary_range=df['salary'].apply(min_max_hourly_wage)) 
df = df.assign(
    median_yearly_salary = df['yearly_salary_range'].apply(
        lambda r: (r[0] + r[1]) / 2 if pd.notnull(r) else r
    ),
    median_hourly_salary = df['hourly_salary_range'].apply(
        lambda r: (r[0] + r[1]) / 2 if pd.notnull(r) else r
    )
)
p=df.groupby(by=['organization'], as_index=False)['median_hourly_salary'].mean().sort_values(by='median_hourly_salary',ascending=False).head()
df.groupby(by=['organization'], as_index=False)['median_yearly_salary'].mean().sort_values(by='median_yearly_salary',ascending=False).head()
df=df[df['sector'].notnull()]
df=df[df['sector'].apply(lambda x: len(x)<35)]
jobtitle=df['job_title'].str.split('Job')
df['job_title']=jobtitle.str[0]
df['sector'].value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True).head()
df['location'].value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True).head()
df['job_title'].value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True).head()
df['organization'].value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True).head()
df.groupby(by=['location'], as_index=False)['organization'].count().sort_values(by='organization',ascending=False).head()
df.groupby(by=['location'], as_index=False)['job_title'].count().sort_values(by='job_title',ascending=False).head()
df.groupby(by=['location'], as_index=False)['sector'].count().sort_values(by='sector',ascending=False).head()