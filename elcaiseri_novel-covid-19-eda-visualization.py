import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns; sns.set()



import warnings

warnings.filterwarnings("ignore")





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
BATH = '/kaggle/input/novel-corona-virus-2019-dataset/'



df = pd.read_csv(BATH + 'covid_19_data.csv', date_parser=['Last Update'])

df.head()
df.info()
main_cols = ['Confirmed', 'Deaths', 'Recovered']

df['Last Update'] = pd.to_datetime(df['Last Update'], unit='ns').dt.date

new_df = df.groupby('Last Update')[main_cols].sum()

new_df.tail(10)
plt.figure(figsize=(20, 8))

plt.plot(new_df['Confirmed'], 'y', lw=4, label='Confirmed')

plt.plot(new_df['Recovered'], 'g', lw=4, label='Recovered')

plt.plot(new_df['Deaths'], 'r', lw=4, label='Deaths')



plt.xlabel('Date', fontsize=20)

plt.ylabel('No. of cases', fontsize=20)

plt.title('Total Cases From 22/1', fontsize=20)



x = new_df.index

labels = x.values

plt.xticks(x, labels, rotation='45')



plt.legend(loc='best')

plt.show()
plt.figure(figsize=(20, 6))



death_rate = (new_df['Deaths'] / new_df['Confirmed']) * 100

plt.plot(death_rate, 'r', lw=4, label='Deaths Rate')

plt.plot(death_rate, 'ko', lw=8)



plt.xlabel('Date', fontsize=20)

plt.ylabel('No. of Deaths Rate %', fontsize=20)

plt.title('Total Cases From 22/1', fontsize=20)



x = new_df.index

labels = x.values

plt.xticks(x, labels, rotation='45')



plt.legend(loc='best')

plt.show()
plt.figure(figsize=(20, 6))



recv_rate = (new_df['Recovered'] / new_df['Confirmed']) * 100

plt.plot(recv_rate, 'g', lw=4, label='Recovered Rate')

plt.plot(recv_rate, 'ko', lw=8)



plt.xlabel('Date', fontsize=20)

plt.ylabel('No. of Recovered Rate %', fontsize=20)

plt.title('Total Cases From 22/1', fontsize=20)



x = new_df.index

labels = x.values

plt.xticks(x, labels, rotation='45')



plt.legend(loc='best')

plt.show()
countrys = df[df['Country/Region'] !='Mainland China'].groupby('Country/Region')[main_cols].sum()

countrys.head(10)
plt.figure(figsize=(25, 10))

countrys[countrys['Confirmed'] > 100000].plot(figsize=(25, 10), kind='bar')

plt.title("Countries that Confirmed more than 100K Cases.", fontsize=20)

plt.xlabel('Countries', fontsize=20);

plt.ylabel('No. of Cases', fontsize=20);

plt.xticks(rotation='45', fontsize=20);

plt.legend()

plt.show()
plt.figure(figsize=(18, 12))

m = countrys[countrys['Confirmed'] > 200000]['Confirmed']

plt.pie(m, shadow=True, autopct='%1.1f%%')

plt.title("Countries that Confirmed more than 200K Cases.", fontsize=20)

plt.xlabel('Countries', fontsize=20);

plt.ylabel('No. of Cases', fontsize=20);

plt.xticks(rotation='30', fontsize=20);

plt.legend(m.index, loc='best')

plt.show()
plt.figure(figsize=(18, 12))

m = countrys[countrys['Deaths'] > 10000]['Deaths']

plt.pie(m, shadow=True, autopct='%1.1f%%')

plt.title("Countries that Deaths more than 10K Cases.", fontsize=20)

plt.xlabel('Countries', fontsize=20);

plt.ylabel('No. of Cases', fontsize=20);

plt.xticks(rotation='30', fontsize=20);

plt.legend(m.index, loc='best')

plt.show()
plt.figure(figsize=(25, 10))

countrys[countrys['Deaths'] > 100]['Deaths'].plot(kind='bar')

plt.title("Countries Deaths more than 100 Cases.", fontsize=20)

plt.xlabel('Countries', fontsize=20);

plt.ylabel('No. of Cases', fontsize=20);

plt.xticks(rotation='90', fontsize=20);
plt.figure(figsize=(25, 10))

countrys[countrys['Recovered'] > 100]['Recovered'].plot(kind='bar')

plt.title("Countries that Recovered more than 100 Cases.", fontsize=20)

plt.xlabel('Countries', fontsize=20);

plt.ylabel('No. of Cases', fontsize=20);

plt.xticks(rotation='90', fontsize=15);
m = countrys[countrys['Confirmed'] > 10000]['Deaths'].sort_values(ascending=False)

plt.figure(figsize=(15, 12))

sns.barplot(x=m, y=m.index)

plt.title("Countries that Confirmed Deaths more than 10K Cases.", fontsize=20);
m = countrys[countrys['Confirmed'] > 10000]['Recovered'].sort_values(ascending=False)

plt.figure(figsize=(15, 12))

plt.title("Countries that Confirmed Recovered more than 10K Cases.", fontsize=20);

sns.barplot(x=m, y=m.index)
china_ = df[df['Country/Region'] == 'Mainland China']

new_china_ = china_.groupby('Last Update')[main_cols].sum()

new_china_.head(10)
plt.figure(figsize=(20, 8))

plt.plot(new_china_['Confirmed'], 'y--', lw=4, label='China Confirmed')

plt.plot(new_china_['Recovered'], 'g--', lw=4, label='China Recovered')

plt.plot(new_china_['Deaths'], 'r--', lw=4, label='China Deaths')



plt.xlabel('Date', fontsize=20)

plt.ylabel('No. of China Cases', fontsize=20)

plt.title('Total China Cases From 22/1', fontsize=20)



x = new_df.index

labels = x.values

plt.xticks(rotation='45')



plt.legend(loc='best')

plt.show()
china = df[df['Country/Region'] == 'Mainland China'].groupby('Province/State')[main_cols].sum()

china.head(10)
m = china['Confirmed'].sort_values(ascending=False)

plt.figure(figsize=(12, 8))

plt.title("China State Confirmed Cases.", fontsize=20);

sns.barplot(x=m, y=m.index)
plt.figure(figsize=(25, 10))

china[china['Deaths'] > 100].plot(figsize=(25, 10), kind='bar')

plt.title("China Deaths more than 100 Cases.", fontsize=20)

plt.xlabel('States', fontsize=20);

plt.ylabel('No. of Cases', fontsize=20);

plt.xticks(rotation='30', fontsize=20);

plt.legend()

plt.show()
italy = df[df['Country/Region'] == 'Italy']

new_italy_ = italy.groupby('Last Update')[main_cols].sum()



plt.figure(figsize=(15, 8))

plt.plot(new_italy_['Confirmed'], 'y--', lw=4, label='Italy Confirmed')

plt.plot(new_italy_['Recovered'], 'g--', lw=4, label='Italy Recovered')

plt.plot(new_italy_['Deaths'], 'r--', lw=4, label='Italy Deaths')



plt.xlabel('Date', fontsize=20)

plt.ylabel('No. of Italy Cases', fontsize=20)

plt.title('Total Italy Cases Between 22/1 and 18/3', fontsize=20)



x = new_df.index

labels = x.values

plt.xticks(rotation='45')



plt.legend(loc='best')

plt.show()
italy
italy.plot(figsize=(18, 5), kind='bar')

plt.title("Total Italy Cases")

plt.ylabel("NO. of Cases.");
data = pd.read_csv(BATH + 'COVID19_line_list_data.csv')

data = data[['id', 'location', 'country', 'gender', 'age', 'death', 'recovered']]

data.head(10)
plt.figure(figsize=(15, 6))

sns.distplot(data['age'], rug=False, bins=50, color='g')

plt.title('Age Distribution')

plt.xlabel("Age");

plt.show()
fig, ax = plt.subplots(1, 2, figsize=(15, 6))



data_ = data[(data['death'] == '1') | (data['death'] == '0')].reset_index()

sns.countplot(x="gender", data=data_, hue='death', ax=ax[0])

ax[0].legend(['Alive', 'Dead'])



data_ = data[(data['recovered'] == '1') | (data['recovered'] == '0')].reset_index()

sns.countplot(x="gender", data=data_, hue='recovered', ax=ax[1])

ax[1].legend(['Dead', 'Recovered'])