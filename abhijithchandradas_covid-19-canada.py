import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df_cases=pd.read_csv("../input/uncover/UNCOVER/covid_19_canada_open_data_working_group/individual-level-cases.csv")

df_mort=pd.read_csv("../input/uncover/UNCOVER/covid_19_canada_open_data_working_group/individual-level-mortality.csv")

print("Shape of Cases df :", df_cases.shape)

print("Shape of Deaths df :", df_mort.shape)

df_cases.head()
df_mort.head()
df_mort.sex.unique()
plt.figure(figsize=(10,4))

plt.suptitle("Gender wise distribution", fontsize=16)

plt.subplot(1,2,1)

plt.title("Confirmed Cases")

sns.countplot('sex', data=df_cases[df_cases.sex!='Not Reported'])



plt.subplot(1,2,2)

plt.title("Deaths Reported")

sns.countplot('sex', data=df_mort[df_mort.sex!='Not Reported'])



plt.show()
df_cases.age.unique()
df_cases.age[df_cases.age=='<18']='10-19'

df_cases.age[df_cases.age=='<1']='0-9'

df_cases.age[df_cases.age=='2']='0-9'

df_cases.age[df_cases.age=='<10']='0-9'

df_cases.age[df_cases.age=='61']='60-69'

df_cases.age[df_cases.age=='50']='50-59'

df_cases.age[df_cases.age=='<20']='10-19'

df_cases.age.value_counts()
df_mort.age.unique()
df_mort.age[df_mort.age=='82']='80-89'

df_mort.age[df_mort.age=='>70']='70-79'

df_mort.age[df_mort.age=='83']='80-89'

df_mort.age[df_mort.age=='78']='70-79'

df_mort.age[df_mort.age=='92']='90-99'

df_mort.age[df_mort.age=='>80']='80-89'

df_mort.age[df_mort.age=='>50']='50-59'

df_mort.age[df_mort.age=='>65']='60-69'

df_mort.age[df_mort.age=='61']='60-69'

df_mort.age.value_counts()
order_age=['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99','100-109']



plt.figure(figsize=(10,4))

plt.suptitle("Age wise distribution")



plt.subplot(1,2,1)

plt.title("Confirmed Cases")

sns.countplot('age', data=df_cases[df_cases.age!='Not Reported'], order=order_age)

plt.xticks(rotation=60)



plt.subplot(1,2,2)

plt.title("Deaths Reported")

sns.countplot('age', data=df_mort[df_mort.age!='Not Reported'], order=order_age)

plt.xticks(rotation=60)



plt.show()
df_cases.province.unique()
order_conf=df_cases[df_cases.province!='Repatriated'].province.value_counts().index
df_mort.province.unique()
df_mort.province.value_counts()
order_mort=df_mort.province.value_counts().index
plt.figure(figsize=(10,4))

plt.suptitle("Province wise distribution")



plt.subplot(1,2,1)

plt.title("Confirmed Cases")

sns.countplot('province', data=df_cases[df_cases.province!='Repatriated'], order=order_conf)

plt.xticks(rotation=60)



plt.subplot(1,2,2)

plt.title("Deaths Reported")

sns.countplot('province', data=df_mort, order=order_mort)

plt.xticks(rotation=60)



plt.show()
pr_mort=pd.DataFrame(df_mort.province.value_counts())

pr_mort.rename(columns={"province":"deaths"}, inplace=True)

pr_mort["cases"]=0

pr_mort["cfr"]=0

for pr in pr_mort.index:

    pr_mort.cases[pr_mort.index==pr]=df_cases.province.value_counts()[pr]

pr_mort.cfr=round(pr_mort.deaths*100/pr_mort.cases,2)

pr_mort.sort_values(by='cfr', ascending=False, inplace=True)



plt.figure(figsize=(8,4))

plt.title("Province wise Case Fatality Ratio")

sns.barplot(y=pr_mort.index, x='cfr', data=pr_mort, orient='h')

plt.show()
plt.figure(figsize=(14,5))

plt.suptitle("Transmission and Travel Histroy", fontsize=16)



plt.subplot(1,2,1)

plt.title("Imported vs Locally Acquired", fontsize=16)

label=["Imported Cases", 'Locally Acquired']

x=[df_cases.travel_yn.value_counts()['1'],df_cases.travel_yn.value_counts()['0']]

plt.pie(x, labels=label, autopct='%1.1f%%')



plt.subplot(1,2,2)

plt.title("Local Transmission", fontsize=16)

x=[df_cases.locally_acquired.value_counts().sum()-df_cases.locally_acquired.value_counts()['Community'],

   df_cases.locally_acquired.value_counts()['Community']]

labels=["Close Contact", "Community Spread"]

plt.pie(x,labels=labels,autopct='%1.1f%%')



plt.show()
imported=df_cases[df_cases.travel_history_country!='Not Reported'].travel_history_country.value_counts()

x=list(imported[imported>10])

y=list(imported[imported>10].index)

x.append(imported[imported<11].sum())

y.append("Others")

sns.barplot(x, y, orient='h')

plt.show()