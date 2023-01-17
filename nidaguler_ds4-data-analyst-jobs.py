import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import re

import nltk

import nltk as nlp

import warnings 

warnings.filterwarnings("ignore")
df=pd.read_csv("../input/data-analyst-jobs/DataAnalyst.csv")
df.head()
df.tail()
df.drop(["Unnamed: 0","Job Description"],axis=1,inplace=True)
df.head()
df.columns
df.rename(columns={'Job Title':'job_title','Salary Estimate':"salary_estimate",'Rating':'rating',

       'Company Name':'company_name', 'Location':'location', 'Headquarters':'headquarters', 'Size':"size", 'Founded':"founded",

       'Type of ownership':'type_of_ownership', 'Industry':'industry', 'Sector':'sector', 'Revenue':'revenue', 'Competitors':'competitors',

       'Easy Apply':'easy_appy'},inplace=True)
df.head()
df.job_title.unique()
df.isna().sum()
if df.empty:

    print('DataFrame is empty!')
df.isnull().values.any()
nan_rows = df[df['company_name'].isnull()]

nan_rows
df.drop([1860],inplace=True)
df.isna().sum()
df.head()
df.salary_estimate.unique()
df["salary_estimate"]=df["salary_estimate"].apply(lambda x:str(x).replace(' (Glassdoor est.)','')if ' (Glassdoor est.)' in str(x) else str(x))

#df["job_title"]=df["job_title"].apply(lambda x: float(x))
df.salary_estimate.unique()
df["salary_estimate"]=df["salary_estimate"].apply(lambda x:str(x).replace('$','')if '$' in str(x) else str(x))
df.salary_estimate.unique()
df["salary_estimate"]=df["salary_estimate"].apply(lambda x:str(x).replace('K','')if 'K' in str(x) else str(x))
df.salary_estimate.unique()
nan_rows = df[df['salary_estimate']=="-1"] 

nan_rows
df.drop([2149],inplace=True)
df["salary_estimate"]=df["salary_estimate"].apply(lambda x: str(x).replace('-',',') if '-' in str(x) else str(x))
df.head()
df.head()
new = df["job_title"].str.split(",", n = 1, expand = True) 

# making separate first name column from new data frame 

df["job"]= new[0] 

  

# making separate last name column from new data frame 

df["last_job"]= new[1] 

  

# Dropping old Name columns 

df.drop(columns =["job_title"], inplace = True) 

df.drop(columns =["last_job"], inplace = True) 

  

# df display 

df.head()
df[['first_salary','last_salary']] = df.salary_estimate.str.split(",",expand=True,)
df.drop(["salary_estimate"],axis=1, inplace=True)
df.info()
df.first_salary=df.first_salary.astype(int)

df.last_salary=df.last_salary.astype(int)
df["salary"]=(df.first_salary+df.last_salary)/2
df.drop(["first_salary","last_salary"],axis=1,inplace=True)
df.head()
df.head()
df.company_name.unique()
df["company_name"]=df["company_name"].apply(lambda x: str(x).replace('\n3.2',' ') if '\n3.2' in str(x) else str(x))

df["company_name"]=df["company_name"].apply(lambda x: str(x).replace('\n3.8',' ') if '\n3.8' in str(x) else str(x))

df["company_name"]=df["company_name"].apply(lambda x: str(x).replace('\n3.4',' ') if '\n3.4' in str(x) else str(x))

df["company_name"]=df["company_name"].apply(lambda x: str(x).replace('\n2.9',' ') if '\n2.9' in str(x) else str(x))

df["company_name"]=df["company_name"].apply(lambda x: str(x).replace('\n3.1',' ') if '\n3.1' in str(x) else str(x))

df["company_name"]=df["company_name"].apply(lambda x: str(x).replace('\n3.4',' ') if '\n3.4' in str(x) else str(x))

df["company_name"]=df["company_name"].apply(lambda x: str(x).replace('\n4.1',' ') if '\n4.1' in str(x) else str(x))

df["company_name"]=df["company_name"].apply(lambda x: str(x).replace('\n3.9',' ') if '\n3.9' in str(x) else str(x))

df["company_name"]=df["company_name"].apply(lambda x: str(x).replace('\n2.5',' ') if '\n2.5' in str(x) else str(x))

df.head()
df["size"]=df["size"].apply(lambda x:str(x).replace('$','')if '$' in str(x) else str(x))
df["size"]=df["size"].apply(lambda x:str(x).replace('to',",")if 'to' in str(x) else str(x))

df["size"]=df["size"].apply(lambda x:str(x).replace('employees','')if "employees" in str(x) else str(x))

df["size"]=df["size"].apply(lambda x:str(x).replace('+','')if '+' in str(x) else str(x))

df.head()
#df[['first_employee_num','second_employee_num']]=df.size.str.split(',',expand=True)