import pandas as pd 

import numpy as np

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('dark')

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

print('Setup complete')
df = pd.read_csv('..//input//covid19-patient-precondition-dataset//covid.csv', index_col='id')
plt.subplots(figsize=(12, 10))

sns.heatmap(df.corr())
ELD = np.zeros_like(df['diabetes'].values, dtype='int32')



for col in df.columns[9:19]:

    uniques = df[col].unique()

    uniques = np.sort(uniques)

    ELD += df[col].replace(uniques[1:], 0).values



df['ELD_indx'] = ELD

df = df.drop(df.columns[9:19], axis=1)



plt.subplots(figsize=(12, 10))

sns.heatmap(df.corr())
no_icu_data_bool = df['icu'].isin([97, 98, 99])

no_icu_data_bool



icu_data = df[~ no_icu_data_bool]

no_icu_data = df[no_icu_data_bool]

print("{} rows have ICU details ".format(icu_data.shape[0]))

print("Only {}% of given data has ICU details ".format(round((icu_data.shape[0]/ no_icu_data.shape[0])*100)))
icu_data.sex.replace({1: 'Female', 2: 'Male'}, inplace=True)

icu_data.patient_type.replace({1: 'Outpatient', 2: 'Inpatient'}, inplace=True)

icu_data.intubed.replace({1: 'Yes', 2: 'No',97:'Not Specified', 98:'Not Specified',99:'Not Specified'}, inplace=True)

icu_data.pneumonia.replace({1: 'Yes', 2: 'No', 98:'Not Specified',99:'Not Specified', 97:'Not Specified'}, inplace=True)

icu_data.pregnancy.replace({1: 'Yes', 2: 'No', 99:'Not Specified',98:'Not Specified', 97:'Not Specified'}, inplace=True)

icu_data.contact_other_covid.replace({1: 'Yes', 2: 'No', 97:'Not Specified',99:'Not Specified',98:'Not Specified'}, inplace=True)

icu_data.covid_res.replace({1: 'Positive', 2: 'Negative', 3:'Awaiting Results'}, inplace=True)

icu_data.icu.replace({1: 'Yes', 2: 'No', 97:'Not Specified',98:'Not Specified', 99:'Not Specified'}, inplace=True)

icu_data.head()
from datetime import datetime

def convert_date(day, first_day="01-01-2020", sep='-'):

    d1 = first_day.replace('-', sep)

    fmt = f'%d{sep}%m{sep}%Y'

    d1 = datetime.strptime(d1, fmt)

    d2 = datetime.strptime(day, fmt)

    delta = d2 - d1

    return delta.days
icu_data['date_died'] = icu_data['date_died'].replace('9999-99-99', 0)

icu_data['day_died'] = icu_data['date_died'].apply(lambda date: np.NaN if date == 0 else convert_date(date))



icu_data['entry_date'] = icu_data['entry_date'].replace('9999-99-99', 0)

icu_data['entry_day'] = icu_data['entry_date'].apply(lambda date: np.NaN if date == 0 else convert_date(date))



icu_data['date_symptoms'] = icu_data['date_symptoms'].replace('9999-99-99', 0)

icu_data['day_symptoms'] = icu_data['date_symptoms'].apply(lambda date: np.NaN if date == 0 else convert_date(date))
icu_data['died'] = icu_data['date_died'].apply(lambda x: 'Non-died' if x == 0 else 'Died')
icu_data
df = icu_data
fig, axarr = plt.subplots(1, 2, figsize=(12,6))

axarr[0].set_title('Age distribution')

f = sns.distplot(df['age'], color='g', bins=40, ax=axarr[0])

axarr[1].set_title('age distribution for the two subpopulations')

g = sns.kdeplot(df['age'].loc[df['died'] == 'Died'], 

                shade= True, ax=axarr[1], label='Died').set_xlabel('Age')

g = sns.kdeplot(df['age'].loc[df['died'] == 'Non-died'], 

                shade=True, ax=axarr[1], label='Not died')
from scipy import stats



mask = df['died'] == 'Died'

died = df['age'][mask]



mask = df['died'] == 'Non-died'

nondied = df['age'][mask]



res = stats.ttest_ind(died, nondied, equal_var=False)

print('p-value:', res[1])
died.describe()
nondied.describe()
print('Number of people for each ELD index value:')

print(df.ELD_indx.value_counts())



fig, ax = plt.subplots(figsize=(12, 8))

ax = sns.barplot(x=df['ELD_indx'].value_counts().keys(),

            y=df['ELD_indx'].value_counts().values)

ax.set_xticklabels(ax.get_xticklabels(), rotation=35)

plt.title('Number of people for each ELD index value')

plt.xlabel('ELD index')

plt.grid()

plt.show()
unique_ELD = np.sort(df['ELD_indx'].unique())

all_ELD = []

died_ELD = []

percentage = []

for indx in unique_ELD:

    all_ELD.append(df['ELD_indx'][df['ELD_indx'] == indx].count())

    died_ELD.append(df['ELD_indx'][(df['ELD_indx'] == indx) & (df['died'] == 'Died')].count())

    percentage.append((died_ELD[-1] / all_ELD[-1]) * 100)



fig, ax = plt.subplots(figsize=(12, 8))

ax = sns.barplot(x=unique_ELD, y=percentage)

ax.set_xticklabels(ax.get_xticklabels(), rotation=35)

plt.title('The percentage of deaths when there is a tendency to lung diseases.')

plt.ylabel('Percentage')

plt.xlabel('ELD index')

plt.grid()

plt.show()
fig, ax = plt.subplots(figsize=(10, 8))

p = sns.countplot(x ='pregnancy', hue ='died', data = df[:][df['sex'] == 'Female'], 

                  ax=ax).set_title('The mortality of pregnant and non-pregnant women')
all_preg_female = df['sex'][(df['sex'] == 'Female') & (df['pregnancy'] == 'Yes')]

died_preg_female = df['sex'][(df['sex'] == 'Female') & (df['pregnancy'] == 'Yes') & (df['died'] == 'Died')]



all_notpreg_female = df['sex'][(df['sex'] == 'Female') & (df['pregnancy'] == 'No')]

died_notpreg_female = df['sex'][(df['sex'] == 'Female') & (df['pregnancy'] == 'No') & (df['died'] == 'Died')]



percentage = round(died_preg_female.count() / all_preg_female.count() * 100, 3)

print(f'Percentage of pregnant women who died: {percentage} %')

percentage = round(died_notpreg_female.count() / all_notpreg_female.count() * 100, 3)

print(f'Percentage of non-pregnant women who died.: {percentage} %')
percentage_f = df['sex'][(df['sex'] == 'Female') & (df['died'] == 'Died')].count() / df['sex'][(df['sex'] == 'Female')].count()

percentage_f = round(percentage_f * 100, 3)



percentage_m = df['sex'][(df['sex'] == 'Male') & (df['died'] == 'Died')].count() / df['sex'][(df['sex'] == 'Male')].count()

percentage_m = round(percentage_m * 100, 3)



fig, ax = plt.subplots(figsize=(10, 8))

ax = sns.barplot(x=['Male', 'Female'], y=[percentage_m, percentage_f])

plt.title('Percentage of deaths among men and women.')

plt.ylabel('Percentage')

plt.xlabel('Gender')

plt.grid()

plt.show()
fig, ax = plt.subplots(figsize=(10, 8))

p = sns.countplot(x ='ELD_indx', hue ='sex', data = df, 

                  ax=ax).set_title('Number of men and women with an ELD index')

plt.grid()
percentage_f = df['sex'][(df['sex'] == 'Female') & (df['pneumonia'] == 'Yes')].count() / df['sex'][(df['sex'] == 'Female')].count()

percentage_f = round(percentage_f * 100, 3)



percentage_m = df['sex'][(df['sex'] == 'Male') & (df['pneumonia'] == 'Yes')].count() / df['sex'][(df['sex'] == 'Male')].count()

percentage_m = round(percentage_m * 100, 3)



fig, ax = plt.subplots(figsize=(10, 8))

ax = sns.barplot(x=['Male', 'Female'], y=[percentage_m, percentage_f])

plt.title('The average age of men and women with pneumonia.')

plt.ylabel('Percentage')

plt.xlabel('Gender')

plt.grid()

plt.show()
def plot_day_counts(data, columns_names, color=None, show_friday=True, figsize=(12, 5)):

    fig, axarr = plt.subplots(figsize=figsize)

    for col in columns_names:

        unique_days = np.sort(data[col].unique())

        unique_days = unique_days[:-1]

        counts = []

        all_days = np.linspace(0.0, unique_days.max(), int(unique_days.max()) + 1)

        for day in all_days:

            counts.append(data[col][(data[col] == day) & (data['ELD_indx'] == 0)].count())

        label = f'{col} counts'

        if color:

            plt.plot(all_days, counts, label=label, color=color)

        else:

            plt.plot(all_days, counts, label=label)

    if show_friday:

        plt.vlines([i for i in range(2, int(unique_days.max()) + 1, 7)], 0, 

                   max(counts), linestyles='--', color='green', alpha=0.5, label='Fridays')

    plt.grid()

    plt.legend()

    plt.xlabel('Number of the day from the beginning of 2020.')

    plt.ylabel('Number of people.')



plot_day_counts(df, ['entry_day'])

plt.title('The number of patients admitted for treatment over time.')
plot_day_counts(df, ['day_symptoms'], color='brown')

plt.title('The number of people who show symptoms over time.')
plot_day_counts(df, ['day_died'], color='red', show_friday=False)

plt.title('The number of deaths over time.')