import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1", low_memory=False)

br_data = data[data.Country == 'Brazil']
print("%s brasileiros responderam a pesquisa do Kaggle" % br_data.shape[0])
counts = data.Country.value_counts().head(10)

counts.plot(kind='pie', figsize=(8,8), autopct='%1.1f%%');

plt.legend(labels=counts.values, loc="best");
fig, axes = plt.subplots(nrows=1, ncols=2)



brgender = br_data.GenderSelect.value_counts()

brgender.plot(kind='pie', figsize=(15,7), autopct='%1.1f%%', ax=axes[0]);



gender = data[data.Country != 'Brazil'].GenderSelect.value_counts()

gender.plot(kind='pie', figsize=(15,7), autopct='%1.1f%%', ax=axes[1]);
br_data.Age.value_counts(bins=10).plot(kind='bar', figsize=(15,6), rot=40, legend=True);
employ = br_data.EmploymentStatus.value_counts()

employ.plot(kind='pie', figsize=(7,7), autopct='%1.1f%%');

plt.legend(labels=employ.values, loc="best");
jobtitle = br_data.CurrentJobTitleSelect.value_counts()

jobtitle.plot(kind='bar', figsize=(15,6), rot=40, legend=True);
sns.factorplot(y='CurrentJobTitleSelect', hue='TitleFit', kind='count', data=br_data, size=8);
salary = br_data[br_data.CompensationAmount.notnull()]

print('%s pessoas informaram seus salários no subset Brasil.' % salary.shape[0])
salary.loc[:, 'CompensationAmount'] = salary.CompensationAmount.apply(

    lambda x: 0 if (pd.isnull(x) or (x=='-') or (x==0)) else float(x.replace(',',''))

)
def monthly2yearly(salary):

    if salary < 30_000:

        return salary * 12

    return salary



salary = salary[(salary.CompensationAmount < 1_000_000) & (salary.CompensationAmount > 2_000)]

salary.loc[:, 'CompensationAmount'] = salary.CompensationAmount.apply(monthly2yearly)
gender_salary = salary.groupby('GenderSelect').agg({'CompensationAmount':['mean','std']})

gender_salary.plot(kind='bar', figsize=(8,6));
group = salary.groupby('Age')['CompensationAmount'].mean()

group = group.reset_index()

age_salary = group.groupby(pd.cut(group['Age'], np.array([20, 30, 40, 50, 60, 80]))).mean()

age_salary['CompensationAmount'].plot(kind='bar', figsize=(8,6), legend=True);
edu_salary = salary.groupby('FormalEducation').agg({'CompensationAmount': ['mean', 'std']})

edu_salary.plot(kind='barh', figsize=(8, 7)).legend(bbox_to_anchor=(-.3, .1));