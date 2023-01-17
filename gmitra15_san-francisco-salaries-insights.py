import pandas as pd
sal = pd.read_csv('../input/Salaries.csv')
sal['BasePay'] = pd.to_numeric(sal['BasePay'], errors='coerce')
sal['OvertimePay'] = pd.to_numeric(sal['OvertimePay'], errors='coerce')
sal['OtherPay'] = pd.to_numeric(sal['OtherPay'], errors='coerce')
sal['Benefits'] = pd.to_numeric(sal['Benefits'], errors='coerce')
sal.head()
sal.info()
print("Average Base Pay: ${}".format(round(sal['BasePay'].mean(), 2)))
print("The highest base pay is ${}".format(round(sal['BasePay'].max(), 2)))
sal['OvertimePay'].max()
sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()]
sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].min()]
print("The average base pay from 2011-2014 was ${}".format(round(sal['BasePay'].mean(), 2)))
sal.groupby('Year').mean()['BasePay']
print("There were {} unique job titles in this data set.".format(sal['JobTitle'].nunique()))
sal['JobTitle'].value_counts()[:10]
len(sal[sal['Year'] == 2013]['JobTitle'].value_counts()[sal[sal['Year'] == 2013]['JobTitle'].value_counts() == 1])
sum(sal[sal['Year'] == 2013]['JobTitle'].value_counts() == 1)
sal['LenTitle'] = sal['JobTitle'].apply(len)

sal[['LenTitle', 'BasePay']].corr()
sal['LenTitle'] = sal['JobTitle'].apply(len)
sal[['LenTitle', 'OtherPay']].corr()
police_mean = sal[sal['JobTitle'].str.lower().str.contains('police')]['TotalPayBenefits'].mean()
fire_mean = sal[sal['JobTitle'].str.lower().str.contains('fire')]['TotalPayBenefits'].mean()
print("On average, people whose title includes 'police' make ${:,} in total compensation.".format(round(police_mean,2)))
print("On average, people whose title includes 'fire' make ${:,} in total compensation. \n".format(round(fire_mean,2)))

pct_diff = (fire_mean - police_mean) * 100 / police_mean
print("People whose title includes 'fire' have a {:.2f}% higher total compensation than those whose title includes 'police'.".format(pct_diff))