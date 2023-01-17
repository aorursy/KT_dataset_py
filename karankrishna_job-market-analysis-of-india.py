import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = (10, 7)

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))
data=pd.read_csv('../input/naukri_com-job_sample.csv')
data.head()
data['company'].value_counts().head(10)
f,ax=plt.subplots(figsize=(15,5))

data['company'].value_counts().head(10).plot(kind = 'bar')
data['industry'].value_counts().head(10)
f,ax=plt.subplots(figsize=(15,5))

data['industry'].value_counts().head(10).plot(kind = 'bar')
data['jobtitle'].value_counts().head(10)
f,ax=plt.subplots(figsize=(15,5))

data['jobtitle'].value_counts().head(10).plot(kind = 'bar')
data['skills'].value_counts().head(10)
f,ax=plt.subplots(figsize=(15,5))

data['skills'].value_counts().head(10).plot(kind = 'bar')
replacements = {

   'joblocation_address': {

      r'(Bengaluru/Bangalore)': 'Bangalore',

      r'Bengaluru': 'Bangalore',

      r'Hyderabad / Secunderabad': 'Hyderabad',

      r'Mumbai , Mumbai': 'Mumbai',

      r'Noida': 'NCR',

      r'Delhi': 'NCR',

      r'Gurgaon': 'NCR', 

      r'Delhi/NCR(National Capital Region)': 'NCR',

      r'Delhi , Delhi': 'NCR',

      r'Noida , Noida/Greater Noida': 'NCR',

      r'Ghaziabad': 'NCR',

      r'Delhi/NCR(National Capital Region) , Gurgaon': 'NCR',

      r'NCR , NCR': 'NCR',

      r'NCR/NCR(National Capital Region)': 'NCR',

      r'NCR , NCR/Greater NCR': 'NCR',

      r'NCR/NCR(National Capital Region) , NCR': 'NCR', 

      r'NCR , NCR/NCR(National Capital Region)': 'NCR', 

      r'Bangalore , Bangalore / Bangalore': 'Bangalore',

      r'Bangalore , karnataka': 'Bangalore',

      r'NCR/NCR(National Capital Region)': 'NCR',

      r'NCR/Greater NCR': 'NCR',

      r'NCR/NCR(National Capital Region) , NCR': 'NCR'

       

   }

}



data.replace(replacements, regex=True, inplace=True)

y = data['joblocation_address'].value_counts()
most_job_posting_city=data['joblocation_address'].value_counts().head()

f ,ax=plt.subplots(figsize=(15,5))

most_job_posting_city.plot(kind = 'bar')
pay_split = data['payrate'].str[1:-1].str.split('-', expand=True)

pay_split.head()

#remove space in left and right 

pay_split[0] =  pay_split[0].str.strip()

#remove comma 

pay_split[0] = pay_split[0].str.replace(',', '')

#remove all character in two condition

# 1 remove if only character

# 2 if start in number remove after all character

pay_split[0] = pay_split[0].str.replace(r'\D.*', '')

#display 

pay_split[0].head()
#remove space in left and right 

pay_split[1] =  pay_split[1].str.strip()

#remove comma 

pay_split[1] = pay_split[1].str.replace(',', '')

#remove all character in two condition

# 1 remove if only character

# 2 if start in number remove after all character

pay_split[1] = pay_split[1].str.replace(r'\D.*', '')

#display 

pay_split[1].head()
pay_split[0] = pd.to_numeric(pay_split[0], errors='coerce')

pay_split[1] = pd.to_numeric(pay_split[1], errors='coerce')
pay=pd.concat([pay_split[0], pay_split[1]], axis=1, sort=False)
pay.rename(columns={0:'min_pay', 1:'max_pay'}, inplace=True)

pay.head()
data=pd.concat([data, pay], axis=1, sort=False)
data.head()
experience_split = data['experience'].str[0:-1].str.split('-', expand=True)

experience_split.head()
#remove space in left and right 

experience_split[1] =  experience_split[1].str.strip()

#remove comma 

experience_split[1] = experience_split[1].str.replace('yr', '')

#remove all character in two condition

# 1 remove if only character

# 2 if start in number remove after all character

experience_split[1] = experience_split[1].str.replace(r'yr', '')

#display 

experience_split[1].head()
experience_split[0] = pd.to_numeric(experience_split[0], errors='coerce')

experience_split[1] = pd.to_numeric(experience_split[1], errors='coerce')
experience=pd.concat([experience_split[0], experience_split[1]], axis=1, sort=False)
experience.rename(columns={0:'min_experience', 1:'max_experience'}, inplace=True)

experience.head()
data=pd.concat([data, experience], axis=1, sort=False)

data.head()
data['avg_pay']=(data['min_pay'].values + data['max_pay'].values)/2

data['avg_experience']=(data['min_experience'].values + data['max_experience'].values)/2
f,ax=plt.subplots(figsize=(15,5))



sns.stripplot(x='min_experience', y='min_pay', data=data, jitter=True)
f,ax=plt.subplots(figsize=(15,5))

sns.pointplot(x='min_experience', y='min_pay', data=data)
f,ax=plt.subplots(figsize=(15,5))

sns.stripplot(x='max_experience', y='max_pay', data=data, jitter=True)

f,ax=plt.subplots(figsize=(15,5))

sns.pointplot(x='max_experience', y='max_pay', data=data)
sns.pairplot(data, 

             size=5, aspect=0.9, 

             x_vars=["min_experience","max_experience"],

             y_vars=["min_pay"],

             kind="reg")
sns.pairplot(data, 

             size=5, aspect=0.9, 

             x_vars=["min_experience","max_experience"],

             y_vars=["max_pay"],

             kind="reg")
sns.jointplot(x='avg_experience', y='avg_pay', data=data, 

              kind="kde",xlim={0,15}, ylim={0,1000000})
f,ax=plt.subplots(figsize=(15,5))

sns.stripplot(x='avg_experience', y='avg_pay', data=data, jitter=True)
f,ax=plt.subplots(figsize=(15,5))

sns.pointplot(x='avg_experience', y='avg_pay', data=data)


data[['min_pay','industry']].groupby(["industry"]).median().sort_values(by='min_pay',ascending=False).head(10).plot.bar(color='lightgreen')
data[['max_pay','industry']].groupby(["industry"]).median().sort_values(by='max_pay',ascending=False).head(10).plot.bar(color='lightblue')
data[['avg_pay','skills']].groupby(["skills"]).median().sort_values(by='avg_pay',ascending=False).head(10).plot.bar(color='lightgreen')
data[['avg_pay','jobtitle']].groupby(["jobtitle"]).median().sort_values(by='avg_pay',ascending=False).head(10).plot.bar(color='y')