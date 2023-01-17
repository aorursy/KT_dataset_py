import pandas as pd
sal = pd.read_csv("../input/Salaries.csv")
sal.head()
sal.info()
for col in ['BasePay', 'OvertimePay', 'OtherPay', 'Benefits']:
    sal[col] = pd.to_numeric(sal[col], errors='coerce')
type(sal)
sal.size
sal.shape
sal['BasePay'].mean()
sal['OvertimePay'].max()
sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['JobTitle']
sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['TotalPayBenefits']
sal[sal['TotalPayBenefits']== sal['TotalPayBenefits'].max()]
sal[sal['TotalPayBenefits']== sal['TotalPayBenefits'].min()]
sal.groupby('Year').mean()['BasePay']
len(sal['JobTitle'].unique())
sal['JobTitle'].value_counts().head()
sum(sal[sal['Year']==2013]['JobTitle'].value_counts() == 1)
def chief_string(title):
    if 'chief' in title.lower():
        return True
    else:
        return False
sum(sal['JobTitle'].apply(lambda x: chief_string(x)))

from matplotlib import pyplot as plt
%matplotlib inline
import numpy as np
def bar(df, name):
    plt.bar(df.Year, df.TotalPay, color = 'darkorange')
    plt.title('{} Salary Trends'.format(name))
    plt.xticks(np.arange(min(df.Year), max(df.Year)+1, 1.0))
    plt.ylim(min(df.TotalPay)*.99, max(df.TotalPay)*1.01)
    plt.show()
    plt.clf()
all_groups = sal.groupby(['Year'],as_index=False).mean()
bar(all_groups, 'All Jobs')