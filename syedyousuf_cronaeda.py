# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
patients_df=pd.read_csv('/kaggle/input/coronavirusdataset/PatientInfo.csv')
patients_df.head()
patients_df.info()
patients_df.isna().sum()
import seaborn as sns

import matplotlib.pyplot as plt
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,6))

plt.suptitle("Count Plots Of Different Variables Against Gender")

sns.countplot('sex',data=patients_df,ax=ax1)

sns.countplot('country',data=patients_df,ax=ax2,hue='sex')

sns.countplot('state',data=patients_df,ax=ax3,hue='sex')

plt.show()
plt.figure(figsize=(40,6))

plt.title("Infection Case Count Plot Against Gender")

sns.countplot('infection_case',data=patients_df,hue='sex')

plt.show()
patients_df['age']=patients_df['age'].str.replace('s','').astype(float)
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,6))

plt.suptitle('Distributions of Patient Age, Infection_Order, Number of Contact')

sns.distplot(patients_df['age'],ax=ax1,kde=False)

sns.distplot(patients_df['infection_order'],ax=ax2,kde=False)

sns.distplot(patients_df['contact_number'],ax=ax3,kde=False)

plt.show()
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(20,6))



plt.suptitle('Age vs Infection_Order vs Contact_Number vs State')



sns.scatterplot('age','infection_order',data=patients_df,hue='state',ax=ax1)



sns.scatterplot('age','contact_number',data=patients_df,hue='sex',ax=ax2)



sns.barplot('state','age',data=patients_df,hue='sex',ax=ax3)



plt.show()
print(len(patients_df[patients_df['contact_number']==1160.0]),"Patient with most number of Contacts ",patients_df['contact_number'].max())

print(len(patients_df[patients_df['contact_number']==0.0]),"Patients with few number of Contacts ",patients_df['contact_number'].min())
region_df=pd.read_csv("/kaggle/input/coronavirusdataset/Region.csv")
region_df.head()
region_df.isna().sum()
region_df['province'].value_counts()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(25,6))

sns.heatmap(region_df.corr(),annot=True,ax=ax1,cmap='YlGnBu')

sns.heatmap(region_df.describe(),annot=True,ax=ax2,cmap='YlGnBu')