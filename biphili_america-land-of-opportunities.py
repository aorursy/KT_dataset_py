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
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
img=np.array(Image.open('../input/h1bvisa/H1b.jpg'))
fig=plt.figure(figsize=(10,10))
plt.imshow(img,interpolation='bilinear')
plt.axis('off')
plt.ioff()
plt.show()
data=pd.read_csv("../input/h-1b-visa/h1b_kaggle.csv")
data.head()
data.isnull().sum()
data['PREVAILING_WAGE'].fillna((data['PREVAILING_WAGE'].mean()), inplace=True)
plt.figure(figsize=(12,6))
data[data['PREVAILING_WAGE']<200000].PREVAILING_WAGE.hist(bins=50,color='#ffd700')
plt.axvline(data[data['PREVAILING_WAGE']<=200000].PREVAILING_WAGE.mean(),color='black',linestyle='dashed',linewidth=3)
plt.xlabel('Annual Salary in Dollar',fontsize=20)
plt.ylabel('Number of People',fontsize=20)
plt.title('Wage Distribution in Dollar',fontsize=25)

#print('Minumum salary for H1b1 Visa holder is:',int(data[data['PREVAILING_WAGE']<=200000].PREVAILING_WAGE.min()),'$')
print('Mean salary for H1b1 Visa holder is:',int(data[data['PREVAILING_WAGE']<=200000].PREVAILING_WAGE.mean()),'$')
print('Median salary for H1b1 Visa holder is:',int(data[data['PREVAILING_WAGE']<=200000].PREVAILING_WAGE.median()),'$')
#print('Maximum salary for H1b1 Visa holder is:',int(data[data['PREVAILING_WAGE']<=9000000].PREVAILING_WAGE.max()),'$')
data.CASE_STATUS.unique()
data_c = data[data["CASE_STATUS"]=='CERTIFIED' ]
data_c
plt.figure(figsize=(12,8))
data['YEAR'].value_counts().sort_values().plot(marker='o',linewidth=2,color='#ffd700')
plt.xlabel('Year',fontsize=20)
plt.ylabel('Number of Applicants ',fontsize=20)
plt.title('H1B Applicants by Year',fontsize=25)
plt.tick_params(labelsize=15)
plt.grid()
plt.ioff()

plt.xlim([2011,2016])
plt.show()
plt.figure(figsize=(12,8))
plt.hist(data_c["YEAR"].dropna(),bins=18,width=0.5,edgecolor='k',linewidth=2,color='#ffd700')
plt.xlabel('Year',fontsize=20)
plt.ylabel('Number of Certified Applicants ',fontsize=20)
plt.title('H1b Visa Certified Yearwise',fontsize=25)
plt.tick_params(labelsize=15)
plt.grid()
plt.ioff()
plt.figure(figsize=(12,8))
ax=data['CASE_STATUS'].value_counts().sort_values(ascending=True).plot.barh(width=0.9,color='#ffd700',edgecolor='k',linewidth=2)
for i, v in enumerate(data['CASE_STATUS'].value_counts().sort_values(ascending=True).values): 
    ax.text(.2, i, v,fontsize=12,color='r',weight='bold')
plt.title('Case Status for All Years',fontsize=30)
plt.xlabel('Number of Applicants',fontsize=20)
plt.tick_params(labelsize=15)
plt.grid()
plt.ioff()
plt.show()
plt.figure(figsize=(10,8))
ax=data['EMPLOYER_NAME'].value_counts().sort_values(ascending=False)[:10].plot.barh(width=0.8,edgecolor='k',linewidth=2,color='#ffd700')
for i, v in enumerate(data['EMPLOYER_NAME'].value_counts().sort_values(ascending=False).values[:10]): 
    ax.text(.8, i, v,fontsize=12,color='r',weight='bold')
plt.title('Highest Employeer')
fig=plt.gca()
fig.invert_yaxis()
plt.grid()
plt.ioff()
plt.show()
plt.figure(figsize=(10,8))
ax=data['JOB_TITLE'].value_counts().sort_values(ascending=False)[:10].plot.barh(width=0.8,edgecolor='k',linewidth=2,color='#ffd700')
for i, v in enumerate(data['JOB_TITLE'].value_counts().sort_values(ascending=False).values[:10]): 
    ax.text(.8, i, v,fontsize=12,color='r',weight='bold')
plt.title('Skills in Demand')
fig=plt.gca()
fig.invert_yaxis()
plt.grid()
plt.ioff()
plt.show()
data_peeps=data.dropna(subset=['JOB_TITLE'])
data_scientists=data_peeps[data_peeps['JOB_TITLE'].str.contains('DATA SCIENTIST')]
plt.figure(figsize=(10,8))
data_coun=data_scientists['EMPLOYER_NAME'].value_counts()[:10]
ax=sns.barplot(y=data_coun.index,x=data_coun.values,palette=sns.color_palette('inferno',10))
for i, v in enumerate(data_coun.values): 
    ax.text(.5, i, v,fontsize=15,color='white',weight='bold')
plt.title('Companies Hiring Data Scientists')
plt.grid()
plt.ioff()
plt.show()
post1 = data[data['JOB_TITLE'].str.match('DATA SCIENTIST', na=False)]
# filter job title containing "POST DOCTORAL"
post2 = data[data['JOB_TITLE'].str.match('DATA SCIENCE ENGINEER', na=False)]
# filter job title containing "POSTDOC"
post3 = data[data['JOB_TITLE'].str.match('DATA SCIENCES ENGINEER I', na=False)]
# join the three dataframes
datasci = post1.append([post2, post3])
datasci =datasci[datasci['PREVAILING_WAGE']<200000] # Done to remove the outliers in the data

sns.distplot(datasci['PREVAILING_WAGE'])
plt.title('Salary Distribution of Data Scientist')
plt.grid()
plt.ioff()
plt.show()
# show summary stats
print('Mean Salary of Data Scientist is:', datasci['PREVAILING_WAGE'].mean())
print('Median Salary of Data Scientist is:', datasci['PREVAILING_WAGE'].median())
print('Minimum Salary of Data Scientist is:', datasci['PREVAILING_WAGE'].min())
print('Maximum Salary of Data Scientist is:', datasci['PREVAILING_WAGE'].max())

# The mean ($ 117K) looks very high, so I looked at median, which is reasonable. So there must be some 
# very large numbers driving the mean to become so large?
# Looked at the max value, which is absurd for a postdoc salary
#datasci['YEAR'] = datasci['YEAR'].astype(int)
#sns.boxplot(x='YEAR' , y='PREVAILING_WAGE', data = datasci)
plt.rcParams['figure.figsize']=(15,10)
ax = sns.boxplot(x="YEAR", y="PREVAILING_WAGE", data=datasci,width=0.8,linewidth=3)
ax.set_xlabel('Year',fontsize=20)
ax.set_ylabel('Salary in Dollars',fontsize=20)
plt.title('Salary Variation Over Time',fontsize=30)
ax.tick_params(axis='x',labelsize=20,rotation=90)
ax.tick_params(axis='y',labelsize=20,rotation=0)
plt.grid()
plt.ioff()
