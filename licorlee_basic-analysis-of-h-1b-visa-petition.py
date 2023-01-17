import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# define some useful function here

def groupby_and_count(df, col, n_largest=None):

    if n_largest != None:

        return df[[col, 'Unnamed: 0']].groupby(col).count()['Unnamed: 0'].nlargest(n_largest)

    return df[[col, 'Unnamed: 0']].groupby(col).count()['Unnamed: 0']
overview = pd.read_csv('../input/h1b_kaggle.csv')

overview.head()
print('total records:{}'.format(len(overview)))
overview.tail()
overview = overview.dropna()

overview.tail()
startDF = overview.drop(['lon', 'lat'], 1)

startDF['SOC_NAME'] = startDF.SOC_NAME.apply(lambda x: x.upper())

startDF.head()
total_num_by_year = groupby_and_count(startDF, 'YEAR')

total_num_by_year.plot(title='trend of applications')
plt.figure(figsize=(4,4))

groupby_and_count(startDF, 'FULL_TIME_POSITION').plot.pie(title='Full time position overview')
top_career = groupby_and_count(startDF, 'SOC_NAME', 20)

plt.figure(figsize=(16,6))

top_career.plot.bar(title='top 20 occupation of applicants')
salary = startDF[['PREVAILING_WAGE', 'Unnamed: 0']]

salary['SALARY_RANGE'] = salary.PREVAILING_WAGE.apply(np.round, decimals=-4)

salary['SALARY_RANGE'] = salary.SALARY_RANGE.apply(lambda x: x / 10000)

salary.head()
salary_count = groupby_and_count(salary, 'SALARY_RANGE')

salary_count.describe()
# the salary is recognized as reasonable when more than 10 applicants got it

normal_salary = salary.groupby('SALARY_RANGE').count()

normal_salary = normal_salary[normal_salary['Unnamed: 0'] > 10]

normal_salary.describe()
normal_salary['Unnamed: 0'].plot.bar(title='salary range')
normal_salary['Unnamed: 0'][:20].plot.bar()
top_employee = groupby_and_count(startDF, 'EMPLOYER_NAME', 30)

plt.figure(figsize=(16,6))

top_employee[:30].plot.bar(title='top 30 employee sent application')
work_site = groupby_and_count(startDF, 'WORKSITE')

len(work_site)
top_city = work_site.nlargest(10)

plt.figure(figsize=(10,4))

top_city.plot.bar(title='top 10 work sites of applicants')