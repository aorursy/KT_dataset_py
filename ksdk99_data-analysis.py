# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt
dataset=pd.read_csv('../input/jobs-on-naukricom/naukri_com-job_sample.csv')
dataset
dataset.columns
dataset.drop('jobid',inplace=True,axis=1)
dataset.drop('jobdescription',inplace=True,axis=1)
dataset.drop('uniq_id',inplace=True,axis=1)
dataset
dataset.isnull().any()
dataset.drop('postdate',axis=1,inplace=True)
dataset
dataset
companies=dataset['company'].value_counts().head(15)

companies
f,ax=plt.subplots(figsize=(15,5))



companies.plot(kind='bar')
skills=dataset['skills'].value_counts().head(15)

skills
f,ax=plt.subplots(figsize=(15,5))

skills.plot(kind='barh')
industry=dataset['industry'].value_counts().head(15)

industry
f,ax=plt.subplots(figsize=(15,5))

industry.plot(kind='pie')
jobtitle=dataset['jobtitle'].value_counts()
jobtitle.head(15).plot(kind='bar')

joblocations=dataset['joblocation_address'].value_counts()
joblocations
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

dataset.replace(replacements,regex=True,inplace=True)
dataset
joblocations=dataset['joblocation_address'].value_counts()
joblocations.head()
joblocations.head(5).plot(kind='barh')
dataset
jobaddress=pd.DataFrame(dataset['joblocation_address'])

numposition=pd.DataFrame(dataset['numberofpositions'])
positions=pd.concat([jobaddress,numposition],axis=1)
positions
positions.dropna(inplace=True)
positions['joblocation_address'].value_counts()
bangalore_positions=positions[positions['joblocation_address']=='Bangalore']['numberofpositions'].sum()

chennai_positions=positions[positions['joblocation_address']=='Chennai']['numberofpositions'].sum()

ncr_positions=positions[positions['joblocation_address']=='NCR']['numberofpositions'].sum()

hyderabad_positions=positions[positions['joblocation_address']=='Hyderabad']['numberofpositions'].sum()

mumbai_positions=positions[positions['joblocation_address']=='Mumbai']['numberofpositions'].sum()

height=[bangalore_positions,chennai_positions,ncr_positions,hyderabad_positions,mumbai_positions]
y_pos=[1,2,3,4,5]
bars=['Bangalore','Chennai','NCR','Hyderabad','Mumbai']
plt.bar(y_pos, height, color=(0.2, 0.4, 0.6, 0.7))

plt.xticks(y_pos, bars)

plt.show()
