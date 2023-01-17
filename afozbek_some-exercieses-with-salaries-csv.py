import pandas as pd
df = pd.read_csv('../input/Salaries.csv')
df.head()
df.info()
df['BasePay'].head(1000).mean()
df.loc[df['TotalPayBenefits'].idxmax()]['TotalPayBenefits']
df[df['EmployeeName']=='JOSEPH DRISCOLL']['JobTitle']
df[df['EmployeeName']=='JOSEPH DRISCOLL']['TotalPayBenefits']
#df[df['TotalPayBenefits'] == df['TotalPayBenefits'].max()]['EmployeeName']
df.loc[df['TotalPayBenefits'].idxmax()]['EmployeeName']
df.loc[df['TotalPayBenefits'].idxmin()]['EmployeeName']
df.groupby('Year').mean()['TotalPay']
len(df['JobTitle'].unique())
#df['JobTitle'].nunique()
df['JobTitle'].value_counts().head(7) #head default always return 5 top elements
sum(df[df['Year']==2013]['JobTitle'].value_counts() == 1)
def isInclude(title):
    if 'chief' in title.lower():
        return True
    else:
        return False    
sum(df['JobTitle'].apply(lambda x: isInclude(x)))
df['title_len'] = df['JobTitle'].apply(len) 
df[['title_len','TotalPayBenefits']].corr() # No correlation.