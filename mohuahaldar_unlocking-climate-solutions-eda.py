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
main_dir='/kaggle/input/cdp-unlocking-climate-solutions'

dir_list=os.listdir(main_dir)

corp_dir=os.path.join(main_dir,dir_list[0])

sup_dir=os.path.join(main_dir,dir_list[1])

cities_dir=os.path.join(main_dir,dir_list[2])



corp_list=os.listdir(corp_dir)

corp_res_dir=os.path.join(corp_dir,'Corporations Responses')

corp_ques_dir=os.path.join(corp_dir,'Corporations Questionnaires')

corp_dis_dir=os.path.join(corp_dir,'Corporations Disclosing')

ws='Water Security'

cc='Climate Change'
corp_res_cc=os.path.join(corp_res_dir,cc)

os.listdir(corp_res_cc)
cor_cc_20=pd.read_csv(os.path.join(corp_res_cc,'2020_Full_Climate_Change_Dataset.csv'))

cor_cc_20.head()
cor_cc_20.submission_date=pd.to_datetime(cor_cc_20.submission_date)

cor_cc_20.submission_date

cor_cc_20['submission_year']=cor_cc_20.submission_date.dt.year

cor_cc_20['submission_month']=cor_cc_20.submission_date.dt.month

cor_cc_20['submission_day']=cor_cc_20.submission_date.dt.day
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))

cor_cc_20.groupby('organization')['module_name'].count().sort_values(ascending=False)[:20].plot(kind='barh')

plt.xlabel('count of modules')

plt.show()

plt.figure(figsize=(15,10))

cor_cc_20.groupby('submission_month')['response_value'].count().sort_values(ascending=True).plot(kind='barh')

plt.xlabel('countof responses')

plt.show()

plt.figure(figsize=(15,10))

cor_cc_20.groupby('submission_day')['response_value'].count().sort_values(ascending=True).plot(kind='barh')

plt.xlabel('countof responses')

plt.show()

plt.figure(figsize=(15,10))

cor_cc_20.groupby('module_name')['question_number'].count().sort_values(ascending=True).plot(kind='barh')

plt.xlabel('count of questions')

plt.show()

plt.figure(figsize=(25,15))

cor_cc_20.groupby('question_unique_reference')['response_value'].count().sort_values(ascending=True).plot(kind='barh')

plt.xlabel('count of responses')

plt.ylabel('question_unique_reference')

plt.show()
cor_cc_20.question_unique_reference.nunique()
cities_20=pd.read_csv('/kaggle/input/cdp-unlocking-climate-solutions/Cities/Cities Responses/2020_Full_Cities_Dataset.csv')

cities_20.head()
plt.figure(figsize=(20,10))

cities_20.groupby('Question Name').nunique()['Response Answer'].sort_values(ascending=False)[:50].plot(kind='bar')

plt.ylabel('Count of response')

plt.show()
plt.figure(figsize=(20,15))

ax1=plt.subplot(121, aspect='equal')

count_org_per_region=cities_20.groupby('CDP Region')['Organization'].count()

count_org_per_region.plot(kind='pie',ax=ax1,autopct='%1.1f%%')

plt.title('Count of organization by CDP region')

plt.show()