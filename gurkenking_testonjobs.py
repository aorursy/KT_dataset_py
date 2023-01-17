# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series, DataFrame, Panel

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
appl_sample = pd.read_csv('../input/Wuzzuf_Applications_Sample.csv')

appl_sample.columns = ['unique_id', 'user_id', 'id', 'app_date']

job_posts = pd.read_csv('../input/Wuzzuf_Job_Posts_Sample.csv')

#Next steps: join on the datasets http://pandas.pydata.org/pandas-docs/stable/merging.html 

result = appl_sample.merge(job_posts, on='id', how='left')

#dropping unused columns

result = result.drop(['job_description', 'id', 'unique_id', 'user_id', 'city', 'job_industry2', 'job_industry3', 'salary_minimum', 'salary_maximum', 'num_vacancies', 'career_level', 'experience_years', 'post_date', 'views', 'job_requirements', 'payment_period', 'currency'], 1)

#converting date from object to datetime

result['app_date'] = pd.to_datetime(result['app_date'])

result[:3]
result['job_category1'].value_counts()
resultcounted =result.groupby(pd.Grouper(key='app_date', freq='M'))['job_category1'].value_counts().unstack()



#resultcounted[:3]

resultcounted.info()
# removing the less frequent categories, and display only the top ten categories (sorry for the bad code quality)

del resultcounted['Manufacturing/Production/Operations']

del resultcounted['Project/Program Management']

del resultcounted['Education/Training']

del resultcounted['Logistics/Transportation']

del resultcounted['Media/Journalism/Publishing']

del resultcounted['Tourism/Travel']

del resultcounted['Building Construction/Skilled Trades']

del resultcounted['Biotech/R&D/Science']

del resultcounted['Pharmaceutical']

del resultcounted['Management']

del resultcounted['Medical']

del resultcounted['Research']

del resultcounted['Banking']

del resultcounted['Accounting/Finance/Insurance']

#del resultcounted['Business']

del resultcounted['Human Resources']

del resultcounted['Food Services/Hospitality']

del resultcounted['Fashion']

del resultcounted['Sports and Leisure']

del resultcounted['Legal']



resultcounted.plot(figsize=(18,10))