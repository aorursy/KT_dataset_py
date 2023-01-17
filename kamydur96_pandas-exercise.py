import pandas as pd

import sqlite3
con = sqlite3.connect("../input/database.sqlite")

df = pd.read_sql_query('SELECT * FROM Salaries', con)

df
df.head()
df.info()
#since we have str 'Not Provided' wehave to remove that

df['BasePay'].unique()
#locating 'Not Provided'

df.loc[df['BasePay'] == 'Not Provided']
#dreplace 'Not Provided' with zero's

#df.replace('Not Provided',0,inplace = True)
df.drop([148646,148650,148651,148652], axis =0,inplace=True)
#mean of BasePay

df['BasePay'].unique()

#df['BasePay'].mean()
df['OvertimePay'].max()
df.loc[df['EmployeeName'] == 'JOSEPH DRISCOLL'] ['JobTitle']
df[df['EmployeeName'] =='JOSEPH DRISCOLL']['TotalPayBenefits']
df[df['TotalPayBenefits'] == df['TotalPayBenefits'].max()]
df[df['TotalPayBenefits'] == df['TotalPayBenefits'].max()]['EmployeeName']
df[df['TotalPayBenefits'] == df['TotalPayBenefits'].min()]
df[df['TotalPayBenefits'] == df['TotalPayBenefits'].min()]['EmployeeName']
df['Year'].unique()
df.groupby('Year').mean()['TotalPay']
df['JobTitle'].nunique()
df['JobTitle'].value_counts().head(5)
sum(df[df['Year'] ==2013]['JobTitle'].value_counts()==1)
def chiefStr(title):

    if 'chief' in title.lower():

        return True

    else:

        return False

sum(df['JobTitle'].apply(lambda x : chiefStr(x)))
df['title_len'] = df['JobTitle'].apply(len)
df[['title_len','TotalPayBenefits']].corr()