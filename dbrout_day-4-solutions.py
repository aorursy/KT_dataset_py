#first need to merge the rates with the responses and then compute in US dollars

salary=salary.merge(rates,left_on='CompensationCurrency',right_on='originCountry',how='left')

salary['USSalary']=salary['CompensationAmount']*salary['exchangeRate']

import matplotlib.pyplot as plt

import seaborn as sns

plt.subplots(figsize=(15,8))

salary=salary[salary['USSalary']<1000000]

sns.distplot(salary['USSalary'])

plt.title('Salary Distribution',size=15);
sal_coun=salary.groupby('Country')['USSalary'].median().sort_values(ascending=False)[:15].to_frame()



plt.figure(figsize=(10,6))

ax = sns.barplot('USSalary',sal_coun.index,data=sal_coun,palette='RdYlGn')

ax.set_xlim(40000,120000)

ax.set_xlabel('Median US Salary');