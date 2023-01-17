import pandas as pd
#read the csv file

sal = pd.read_csv('../input/sf-salaries/Salaries.csv', na_values='Not Provided')
#see the statistics of the dataset

sal.describe()
#display the first 5 entries

sal.head()
#check how many entries are there, the datatypes distribution and the memory ***148654 records are present in this dataset

sal.info()
#Number of distinct employee names in this dataset

sal['EmployeeName'].unique().shape
#check the datatype of sal

print(type(sal))
#check the datatype of columnns in this dataframe

print(type(sal['JobTitle']))

print(type(sal['Id']))

print(type(sal['BasePay']))
#we can create a slice of required columns as a new dataframe

new_df = sal[['EmployeeName', 'JobTitle', 'TotalPay', 'Year']]
new_df.head()
#what is average base pay

print(pd.to_numeric(sal['BasePay']).mean())
#what is the  highest amount of OvertimePay?

pay = pd.to_numeric(sal['OvertimePay']).max()

print(f'The highest overtimepay is {pay}')
#Find highest base pay amount

print("The highest base pay is ${}".format(round(pd.to_numeric(sal['BasePay']).max(), 2)))
#What is the job title of Joe Lopez ?

sal[sal['EmployeeName'] == 'Joe Lopez']['JobTitle']
#How much does JOSEPH DRISCOLL make (including benefits)? 

sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits']
#select the details of employee with highest pay

sal[sal['TotalPay'] == sal['TotalPay'].max()]
#select the details of employee with least pay

sal[sal['TotalPay'] == sal['TotalPay'].min()]
#What are the top 7 most common jobs

sal['JobTitle'].value_counts().head(7)
#How many unique job titles are present in this dataset?

val = sal['JobTitle'].nunique()

print(f'{val} number of unique jobs are present')
# Get list of categorical variables

s = (sal.dtypes == 'object')

print(type(s))

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
sal['JobTitle'] = sal['JobTitle'].apply(str.upper)

sal['JobTitle'].value_counts()
#Rename the columns

sal.rename(columns = {'JobTitle': 'JOB-ROLE', 'EmployeeName': 'NAME'}, inplace=True)

sal.head()
#How many people have the word Chief in their job title? 

def check_chief(title):

    if 'chief' in title.lower():

        return True

    else:

        return False
sum(sal['JOB-ROLE'].apply(lambda x: check_chief(x)))
#check for correlations

sal['title_len'] = sal['JOB-ROLE'].apply(len)
sal[['title_len','TotalPayBenefits']].corr() # No correlation.