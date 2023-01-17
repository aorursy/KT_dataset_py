import pandas as pd
sal = pd.read_csv("../input/sf-salaries/Salaries.csv")
sal.head()
sal.info()
sal.describe()
sal['BasePay'].mean()
sal['OvertimePay'].max()
sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['JobTitle']
sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['TotalPayBenefits']
max_paid = sal['TotalPayBenefits'].idxmax()

sal.loc[max_paid]

# sal[sal['TotalPayBenefits']== sal['TotalPayBenefits'].max()]
sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].min()]
sal.groupby('Year').mean()['BasePay']
sal.groupby('Year').sum()
sal['JobTitle'].nunique()
sal['JobTitle'].value_counts().head(10)
sum(sal[sal['Year']==2013]['JobTitle'].value_counts() == 1)
sal[sal['Year']==2013]['JobTitle'].value_counts() == 1
def chief_string(title):

    if 'chief' in title.lower().split():

        return True

    else:

        return False
sum(sal['JobTitle'].apply(lambda x:chief_string(x)))
sal['JobTitle'][0]
sal['title_len'] = sal['JobTitle'].apply(len)

sal['title_len']
sal[['title_len','JobTitle']]
sal[['title_len','TotalPay']].corr()