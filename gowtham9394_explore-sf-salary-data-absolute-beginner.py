# import pandas

import pandas as pd
#Read salaries.csv as a dataframe called sal.

sal = pd.read_csv('../input/sf-salaries/Salaries.csv')
# Check head of the dataframe (sal)

sal.head()
#Now we will check the data type for all columns

sal.dtypes
#it looks like few columns are in object, we will convert them into type number type

series_l = ['BasePay', 'OvertimePay', 'OtherPay', 'Benefits']

for series in series_l:

    sal[series] = pd.to_numeric(sal[series], errors = 'coerce')
# now we will check the dtype again

sal.dtypes
# What is the average BasePay.?

sal['BasePay'].mean()
# What is higest amount of OvertimePay in dataset.?

sal['OvertimePay'].max()
# What is the job title of PATRICK GARDNER.?

sal[sal['EmployeeName'] == 'PATRICK GARDNER']["JobTitle"]
# How much does PATRICK GARDNER makes (including benefits.?)

sal[sal['EmployeeName'] == 'PATRICK GARDNER']['TotalPay']
# What is name of the higest paid person (including benefits.?)

sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()]['EmployeeName']
# What is name of the lowest paid person (including benefits.?)

sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].min()]['EmployeeName']
# What was the average (mean) BasePay of all the employees per year.? (2011-2014)

sal.groupby('Year').mean()['BasePay']
# What was the average (mean) Total Benefits of all the employees per year.? (2011-2014)

sal.groupby('Year').mean()['TotalPayBenefits']
# How many unique job titles are there.?

sal['JobTitle'].nunique()
# what are the top 5 monst common jobs.?

sal['JobTitle'].value_counts().head(5)
# How mant job title were reperesented by one person in 2013.?

sum(sal[sal['Year'] == 2013]['JobTitle'].value_counts()==1)
# How many people have cheif in there job title.?

def cheif_string(title):

    if 'chief' in title.lower().split():

        return True

    else:

        return False

sum(sal['JobTitle'].apply(lambda x: cheif_string(x)))
# correlation between lenght of the job title string and salary

sal [ 'title_len'] = sal['JobTitle'].apply(len)
sal[['TotalPayBenefits', 'title_len']].corr()