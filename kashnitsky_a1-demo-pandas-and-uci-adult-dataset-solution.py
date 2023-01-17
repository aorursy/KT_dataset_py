import numpy as np

import pandas as pd
data = pd.read_csv('../input/adult.data.csv')

data.head()
data['sex'].value_counts()
data.loc[data['sex'] == 'Female', 'age'].mean()
float((data['native-country'] == 'Germany').sum()) / data.shape[0]
ages1 = data.loc[data['salary'] == '>50K', 'age']

ages2 = data.loc[data['salary'] == '<=50K', 'age']

print("The average age of the rich: {0} +- {1} years, poor - {2} +- {3} years.".format(

    round(ages1.mean()), round(ages1.std(), 1),

    round(ages2.mean()), round(ages2.std(), 1)))
data.loc[data['salary'] == '>50K', 'education'].unique() # No
for (race, sex), sub_df in data.groupby(['race', 'sex']):

    print("Race: {0}, sex: {1}".format(race, sex))

    print(sub_df['age'].describe())
data.loc[(data['sex'] == 'Male') &

     (data['marital-status'].isin(['Never-married', 

                                   'Separated', 

                                   'Divorced',

                                   'Widowed'])), 'salary'].value_counts()
data.loc[(data['sex'] == 'Male') &

     (data['marital-status'].str.startswith('Married')), 'salary'].value_counts()
data['marital-status'].value_counts()
max_load = data['hours-per-week'].max()

print("Max time - {0} hours./week.".format(max_load))



num_workaholics = data[data['hours-per-week'] == max_load].shape[0]

print("Total number of such hard workers {0}".format(num_workaholics))



rich_share = float(data[(data['hours-per-week'] == max_load)

                 & (data['salary'] == '>50K')].shape[0]) / num_workaholics

print("Percentage of rich among them {0}%".format(int(100 * rich_share)))
for (country, salary), sub_df in data.groupby(['native-country', 'salary']):

    print(country, salary, round(sub_df['hours-per-week'].mean(), 2))
pd.crosstab(data['native-country'], data['salary'], 

           values=data['hours-per-week'], aggfunc=np.mean).T