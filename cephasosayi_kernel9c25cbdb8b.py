import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
print('Setup Completed')
# Reported csv file contains, reported number of cases across the world
file_path = '../input/malaria-dataset/reported_numbers.csv'
reported_data =pd.read_csv(file_path)
reported_data.head()
# ASSESSING DATA
reported_data.info()
reported_data.columns
# filling in missing data with mean() of that column
reported_data.fillna(value={"No. of cases":reported_data["No. of cases"].mean(), "No. of deaths":reported_data["No. of deaths"].mean()}, inplace=True)
reported_data.Year.astype(int)
# check for missing values
print('\n Total number of null values:\n',reported_data.isnull().sum())

reported_data.head(-10)
# Q. Total number of malaria cases reported worldwide 2000 - 2017

print('\n Total Number of Cases reported worldwide:\n ',reported_data['No. of cases'].sum().astype(int))
# Q. Total number of malaria death reported worldwide from 2000 - 2017

print('\n Total Number of deaths reported worldwide:\n ',reported_data['No. of deaths'].sum().astype(int))
# increase of malaria cases

sns.set(rc={'figure.figsize':(14,4)})

sns.set(style='whitegrid')
ax = sns.lineplot(y=reported_data['No. of cases'],x=reported_data['Year'], palette='Blues_d', data=reported_data, label='Cases')
ax.set(ylim=(0,None))
plt.title("Reported Number of Malaria Cases Worldwide (2000 - 2017)", loc='left')

sns.set(rc={'figure.figsize':(14,4)})
sns.set(style='whitegrid')
ax = sns.barplot(y=reported_data['No. of cases'],x=reported_data['WHO Region'], palette='Blues_d', data=reported_data)
plt.title("Reported Number of Malaria Cases Worldwide (2000 - 2017) Region Most affected", loc='left')

# increase of malaria death
# pd.to_numeric('year')
sns.set(rc={'figure.figsize':(14,4)})
sns.set(style='whitegrid')
ax = sns.lineplot(y=reported_data['No. of deaths'],x=reported_data['Year'], palette='Blues_d', data=reported_data, label='Deaths')
plt.title("Reported Number of Malaria Deaths Worldwide (2000 - 2017)", loc='left')


# Q. which region has most number of deaths
sns.set(rc={'figure.figsize':(14,4)})
sns.set(style='whitegrid')
ax = sns.barplot(y=reported_data['No. of deaths'],x=reported_data['WHO Region'], palette='Blues_d', data=reported_data,)
plt.title("Reported Number of Malaria Deaths Region Most Affected (2000 - 2017)", loc='left')


