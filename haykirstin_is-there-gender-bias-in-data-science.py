import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pylab as plt

import seaborn as sb



cp = sb.color_palette()
df_qus = pd.read_csv('../input/schema.csv')

df_multi = pd.read_csv('../input/multipleChoiceResponses.csv',encoding="ISO-8859-1", low_memory=False,thousands=',')

df_free = pd.read_csv('../input/freeformResponses.csv',low_memory=False)

conversion = pd.read_csv('../input/conversionRates.csv')
gender = df_multi['GenderSelect'].value_counts()

ax = gender.plot(kind="bar", figsize=(5,5))
man = df_multi[df_multi['GenderSelect']=='Male'].copy()

woman = df_multi[df_multi['GenderSelect']=='Female'].copy()

fig,axs = plt.subplots(1,2,figsize=(10,4))

sb.distplot(man['Age'].dropna(),ax=axs[1],color=cp[0])

plt.setp(axs[1],title='Male')

sb.distplot(woman['Age'].dropna(),ax=axs[0],color=cp[1])

plt.setp(axs[0],title='Female')
# convert salary information

df_salaries = pd.merge(df_multi,conversion,left_on='CompensationCurrency',right_on='originCountry',how='left')

df_salaries['CompensationAmount'] = df_salaries['CompensationAmount'].replace({'\$': '', ',': ''}, regex=True)

df_salaries['CompensationAmount'] = df_salaries['CompensationAmount'].apply(pd.to_numeric, errors='coerce')

df_salaries['salaryUSD'] = df_salaries['CompensationAmount']*df_salaries['exchangeRate']

df_salaries2 = df_salaries[df_salaries['salaryUSD']<400000]

df_salaries2 = df_salaries2[df_salaries2['salaryUSD']>0]

df_salaries2 = df_salaries2[df_salaries2['GenderSelect'].isin(['Male','Female'])]

df_salaries2.hist(by='GenderSelect',column='salaryUSD',sharex=True,bins=50,figsize=(10,4))
male_satisfaction = man['JobSatisfaction'].value_counts().to_frame()

male_satisfaction['index1'] = male_satisfaction.index



female_satisfaction = woman['JobSatisfaction'].value_counts().to_frame()

female_satisfaction['index1'] = female_satisfaction.index

sorter = ['I prefer not to share','1 - Highly Dissatisfied', '2', '3', '4', '5', '6', '7', '8', '9', '10 - Highly Satisfied']



male_satisfaction['index1'] = male_satisfaction['index1'].astype('category')

male_satisfaction['index1'].cat.set_categories(sorter, inplace=True)

male_satisfaction = male_satisfaction.sort_values(['index1'])



female_satisfaction['index1'] = female_satisfaction['index1'].astype('category')

female_satisfaction['index1'].cat.set_categories(sorter, inplace=True)

female_satisfaction = female_satisfaction.sort_values(['index1'])





fig,axs = plt.subplots(1,2,figsize=(10,5))

my_cp1 = [cp[0] for x in range(0,11)]

my_cp2 = [cp[1] for x in range(0,11)]

axs[1] = male_satisfaction.plot(kind='bar',ax=axs[1],color=my_cp1)

plt.setp(axs[1],title='Male')

axs[0] = female_satisfaction.plot(kind='bar',ax=axs[0],color=my_cp2)

plt.setp(axs[0],title='Female')