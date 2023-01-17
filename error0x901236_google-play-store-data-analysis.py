import os



import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

data_df = pd.read_csv('../input/google-playstore-apps/Google-Playstore-Full.csv')
data_df.sample(10)
data_df.shape
data_df.columns
data_df.info()
PMS_df = data_df.copy()
PMS_df.Category.value_counts().tail(20)
# shifted

strange_data = [' ETEA & MDCAT', ' not notified you follow -', '6', ' Speaker Pro 2019', ' Alfabe �?ren', ' Mexpost)', ' Podcasts', ' Accounting', ' Islamic Name Boy & Girl+Meaning']

a = PMS_df.Category.isin(strange_data)

index = []

for i in range(len(a)):

    if a[i] == True:

        index.append(i)

PMS_df.iloc[index]
# replaced

PMS_df.iloc[index[0], 1:10] = list(PMS_df.iloc[index[0], 4:13])

PMS_df.iloc[index[0], 11:] = np.nan

PMS_df.iloc[index[1],0:10] = list(PMS_df.iloc[index[1], 4:14])

PMS_df.iloc[index[1], 11:] = np.nan

PMS_df.iloc[index[2], 1:10] = list(PMS_df.iloc[index[2], 2:11])

PMS_df.iloc[index[2], 11:] = np.nan

PMS_df.iloc[index[3], 1:10] = list(PMS_df.iloc[index[3], 2:11])

PMS_df.iloc[index[3], 11:] = np.nan

PMS_df.iloc[index[4], 1:10] = list(PMS_df.iloc[index[4], 2:11])

PMS_df.iloc[index[4], 11:] = np.nan

PMS_df.iloc[index[5], 1:10] = list(PMS_df.iloc[index[5], 2:11])

PMS_df.iloc[index[5], 11:] = np.nan

PMS_df.iloc[index[6], 1:10] = list(PMS_df.iloc[index[6], 2:11])

PMS_df.iloc[index[6], 11:] = np.nan

PMS_df.iloc[index[7], 1:10] = list(PMS_df.iloc[index[7], 2:11])

PMS_df.iloc[index[7], 11:] = np.nan

PMS_df.iloc[index[8], 1:10] = list(PMS_df.iloc[index[8], 3:12])

PMS_df.iloc[index[8], 11:] = np.nan



PMS_df.iloc[index]
# shifted

strange_data = [' Channel 2 News', 'Gate ALARM', ' T�rk Alfabesi', ' super loud speaker booster', ' Tour Guide', ' Romantic Song Music Love Songs', ' Breaking News', ')', 'TRAVEL']

a = PMS_df.Category.isin(strange_data)

index = []

for i in range(len(a)):

    if a[i] == True:

        index.append(i)

PMS_df.iloc[index]
# replaced

PMS_df.iloc[index[0],1:10] = list(PMS_df.iloc[index[0], 2:11])

PMS_df.iloc[index[0], 11:] = np.nan

PMS_df.iloc[index[1], 1:10] = list(PMS_df.iloc[index[1], 2:11])

PMS_df.iloc[index[1], 11:] = np.nan

PMS_df.iloc[index[2], 1:10] = list(PMS_df.iloc[index[2], 2:11])

PMS_df.iloc[index[2], 11:] = np.nan

PMS_df.iloc[index[3], 1:10] = list(PMS_df.iloc[index[3], 2:11])

PMS_df.iloc[index[3], 11:] = np.nan

PMS_df.iloc[index[4], 1:10] = list(PMS_df.iloc[index[4], 2:11])

PMS_df.iloc[index[4], 11:] = np.nan

PMS_df.iloc[index[5], 1:10] = list(PMS_df.iloc[index[5], 2:11])

PMS_df.iloc[index[5], 11:] = np.nan

PMS_df.iloc[index[6], 1:10] = list(PMS_df.iloc[index[6], 2:11])

PMS_df.iloc[index[6], 11:] = np.nan

PMS_df.iloc[index[8], 1:10] = list(PMS_df.iloc[index[8], 2:11])

PMS_df.iloc[index[8], 11:] = np.nan



PMS_df.iloc[index]
print(PMS_df['Unnamed: 11'].unique())

print(PMS_df['Unnamed: 12'].unique())

print(PMS_df['Unnamed: 13'].unique())

print(PMS_df['Unnamed: 14'].unique())
# shifted

strange_data = ['4.0.0.0']

a = PMS_df['Unnamed: 11'].isin(strange_data)

index = []

for i in range(len(a)):

    if a[i] == True:

        index.append(i)

PMS_df.iloc[index]
# replaced

PMS_df.iloc[index[0],1:10] = list(PMS_df.iloc[index[0], 2:11])

PMS_df.iloc[index[0], 11:] = np.nan



PMS_df.iloc[index]
# Rating

PMS_df.Rating.value_counts()
PMS_df['Rating'] = pd.to_numeric(PMS_df.Rating, errors='coerce')
# Reviews

PMS_df.Reviews.value_counts()
PMS_df['Reviews'] = pd.to_numeric(PMS_df.Reviews, errors='coerce')
PMS_df.Installs.value_counts()
PMS_df.Installs = PMS_df.Installs.str.replace('+','')

PMS_df.Installs = PMS_df.Installs.str.replace(',','')

PMS_df['Installs'] = pd.to_numeric(PMS_df.Installs, errors='coerce')
PMS_df.Size.value_counts()
PMS_df.Price.value_counts()
PMS_df.Price = PMS_df.Price.str.replace('$','')

PMS_df['Price'] = pd.to_numeric(PMS_df.Price, errors='coerce')
distribution_model = ['Free' if i == 0 else 'Paid' for i in PMS_df['Price']]

PMS_df['Distribution model'] = pd.Series(distribution_model, name = 'Distribution model')
PMS_df['Content Rating'].value_counts()
PMS_df = PMS_df.drop(['Last Updated', 'Minimum Version',

       'Latest Version', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13',

       'Unnamed: 14'],axis=1)
PMS_df
PMS_df.info()
PMS_df.describe()
plt.figure(figsize=(20,5))

plt.title('Category distribution')

PMS_df.Category.value_counts().plot(kind='bar',)

games = PMS_df[PMS_df.Category.str.contains('GAME', regex=False)]

other = PMS_df[~PMS_df.Category.str.contains('GAME', regex=False)]
plt.figure(figsize=(20,5))

plt.title('Games category distribution')

games.Category.value_counts().plot(kind='bar',)
plt.figure(figsize=(20,5))

plt.title('Other apps category distribution')

other.Category.value_counts().plot(kind='bar',)
plt.figure(figsize=(10,10))

plt.title('Ratings distribution')

sns.distplot(PMS_df.Rating[PMS_df.Reviews > 1000], kde=False)
plt.figure(figsize=(8,8))

plt.title('Content Rating distribution')

PMS_df['Content Rating'].value_counts().plot(kind='bar')
rev = 1000000



rat1 = PMS_df.Rating[PMS_df.Reviews <= rev]

rev1 = PMS_df.Reviews[PMS_df.Reviews <= rev]



rat2 = PMS_df.Rating[PMS_df.Reviews > rev]

rev2 = PMS_df.Reviews[PMS_df.Reviews > rev]



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30,8))

fig.suptitle('Reviews affect on the Rating')



ax1.scatter(rev1,rat1)

ax1.set_xlabel('Reviews')

ax1.set_ylabel('Rating')



ax2.scatter(rev2,rat2)

ax2.set_xlabel('Reviews')

ax2.set_ylabel('Rating')



plt.show()
plt.figure(figsize=(20,8))

plt.title('Mean Rating per Category')

plt.grid()

plt.xlabel('Category')

plt.xticks(rotation=90)

plt.ylabel('Rating')



d = PMS_df.groupby('Category')['Rating'].mean().reset_index()

plt.scatter(d.Category, d.Rating)
Dist_method = PMS_df['Distribution model'].value_counts()

plt.figure(figsize=(10,10))

plt.title('Dist model')

plt.pie(Dist_method, labels=Dist_method.index, autopct='%1.1f%%', startangle=180);
plt.figure(figsize=(20,5))

price = PMS_df.Price[PMS_df.Price > 0].value_counts()

(price.head(50)).plot(kind = 'bar')
plt.figure(figsize=(20,5))

size = PMS_df.Size.value_counts(normalize = False)



(size.head(100)).plot(kind = 'bar')