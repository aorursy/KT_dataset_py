# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/uconn_comp_2018_train.csv')
data.head()
data.isnull().sum() #checking for total null values
data.describe()
data_no_minus_1=data.loc[data['fraud']!=-1]
data_no_minus_1.describe()
# 3 rows fraud =-1
f,ax=plt.subplots(1,2,figsize=(18,8))
data_no_minus_1['fraud'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('fraud')
ax[0].set_ylabel('')
sns.countplot('fraud',data=data,ax=ax[1])
ax[1].set_title('fraud')
plt.show

#unbalanced classification
data=data_no_minus_1

data.groupby(['gender','fraud'])['fraud'].count()
f,ax=plt.subplots(1,2,figsize=(18,8))
data[['gender','fraud']].groupby(['gender']).mean().plot.bar(ax=ax[0])
ax[0].set_title('fraud vs gender')
sns.countplot('gender',hue='fraud',data=data,ax=ax[1])
ax[1].set_title('Sex')
plt.show()
data.groupby(['marital_status','fraud'])['fraud'].count()
f,ax=plt.subplots(1,2,figsize=(18,8))
data[['marital_status','fraud']].groupby(['marital_status']).mean().plot.bar(ax=ax[0])
ax[0].set_title('fraud vs marriage')
sns.countplot('marital_status',hue='fraud',data=data,ax=ax[1])
ax[1].set_title('Marriage')
plt.show()
data.groupby(['high_education_ind','fraud'])['fraud'].count()
f,ax=plt.subplots(1,2,figsize=(18,8))
data[['high_education_ind','fraud']].groupby(['high_education_ind']).mean().plot.bar(ax=ax[0])
ax[0].set_title('fraud vs high_education')
sns.countplot('high_education_ind',hue='fraud',data=data,ax=ax[1])
ax[1].set_title('high_education')
plt.show()
data.groupby(['address_change_ind','fraud'])['fraud'].count()
f,ax=plt.subplots(1,2,figsize=(18,8))
data[['address_change_ind','fraud']].groupby(['address_change_ind']).mean().plot.bar(ax=ax[0])
ax[0].set_title('fraud vs address_change')
sns.countplot('address_change_ind',hue='fraud',data=data,ax=ax[1])
ax[1].set_title('address_change')
plt.show()
data.groupby(['witness_present_ind','fraud'])['fraud'].count()
f,ax=plt.subplots(1,2,figsize=(18,8))
data[['witness_present_ind','fraud']].groupby(['witness_present_ind']).mean().plot.bar(ax=ax[0])
ax[0].set_title('fraud vs witness_present')
sns.countplot('witness_present_ind',hue='fraud',data=data,ax=ax[1])
ax[1].set_title('witness_present')
plt.show()
data.groupby(['policy_report_filed_ind','fraud'])['fraud'].count()
f,ax=plt.subplots(1,2,figsize=(18,8))
data[['policy_report_filed_ind','fraud']].groupby(['policy_report_filed_ind']).mean().plot.bar(ax=ax[0])
ax[0].set_title('fraud vs policy_report_filed_ind')
sns.countplot('policy_report_filed_ind',hue='fraud',data=data,ax=ax[1])
ax[1].set_title('policy_report_filed')
plt.show()
data.groupby(['past_num_of_claims','fraud'])['fraud'].count()
f,ax=plt.subplots(1,2,figsize=(18,8))
data[['past_num_of_claims','fraud']].groupby(['past_num_of_claims']).mean().plot.bar(ax=ax[0])
ax[0].set_title('fraud vs past_num_of_claims')
sns.countplot('past_num_of_claims',hue='fraud',data=data,ax=ax[1])
ax[1].set_title('past_num_of_claims')
plt.show()
pd.crosstab(data.zip_code,data.fraud,margins=True).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(18,8))
data[['zip_code','fraud']].groupby(['zip_code']).mean().plot.bar(ax=ax[0])
ax[0].set_title('fraud vs zip_code')
sns.countplot('zip_code',hue='fraud',data=data,ax=ax[1])
ax[1].set_title('zip_code')
plt.show()
pd.crosstab(data.age_of_driver,data.fraud,margins=True).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(18,8))
data[['age_of_driver','fraud']].groupby(['age_of_driver']).mean().plot.bar(ax=ax[0])
ax[0].set_title('fraud vs age_of_driver')
sns.countplot('age_of_driver',hue='fraud',data=data,ax=ax[1])
ax[1].set_title('age_of_driver')
plt.show()
f,ax=plt.subplots(1,2,figsize=(20,8))
sns.distplot(data.safty_rating,ax=ax[0])
ax[0].set_title('safty_rating')
sns.distplot(data.liab_prct,ax=ax[1])
ax[1].set_title('liab_prct')
plt.show()

f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(data.dropna().age_of_vehicle,ax=ax[0])
ax[0].set_title('age_of_vehicle')
sns.distplot(data.dropna().vehicle_price,ax=ax[1])
ax[1].set_title('vehicle_price')
sns.distplot(data.dropna().vehicle_weight,ax=ax[2])
ax[2].set_title('vehicle_weight')
plt.show()
sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()
