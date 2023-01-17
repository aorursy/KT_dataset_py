import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline

df = pd.read_csv('../input/london-stop-and-search.csv', low_memory=False)

df.head(5)
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.info()
df.drop(['policing_operation', 'self-defined_ethnicity', 'part_of_a_policing_operation', 'latitude', 'longitude', 'outcome_linked_to_object_of_search', 'removal_of_more_than_just_outer_clothing'], axis=1, inplace=True)
df.describe()
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].dt.date
df.rename({'officer-defined_ethnicity':'ethnicity'}, axis=1, inplace=True)
df.head()
df.isnull().sum()
df_clean = df.dropna() #remove all null values so we have clean dataset
df_clean.info() #quick check
plot_month = df_clean.groupby('date').size().reset_index(name='number of outcomes').set_index('date')
plot_month.plot(kind='line', figsize=(20,10));
#there is a lot of blank data for 2015, I am going to remove this from the dataframe
df['date'] = pd.to_datetime(df['date'])
df_clean = df[df['date'].dt.year != 2015]
df_clean = df_clean.dropna()
count_by_date = df_clean.groupby('date').size()

#z = np.polyfit(x, 1)
#p = np.poly1d(z)

plt.figure(figsize=(20,10))
plt.xlabel('Date')
plt.ylabel('Count of Stop and Search Record')
plt.title('Time series graph showing stop and search activity')
plt.plot(count_by_date);
df_clean['ethnicity'].value_counts()
df_clean['ethnicity'].value_counts().plot.bar(title='stop_and_search_by_ethnicity',figsize=(15,10));
df_clean['object_of_search'].value_counts().plot.bar(title='Stop and Search by Object of Search',figsize=(15,10));
df_clean.groupby('ethnicity')['object_of_search'].value_counts().unstack(0).plot.bar(title='Object of search by Ethnicity', figsize=(15,10));
df_clean.groupby('object_of_search')['ethnicity'].value_counts().unstack(0).plot.bar(title='Ethnicity by Object of Search', figsize=(15,10));
counts = df_clean['object_of_search'].value_counts()
counts
counts[counts > 5000]
drugs = df_clean.object_of_search == 'Controlled drugs'
criminal_damage = df_clean.object_of_search == 'Articles for use in criminal damage'
drugs_p_crimdam = df_clean.object_of_search[criminal_damage].count() + df_clean.object_of_search[criminal_damage].count()
drugs_prc = df_clean.object_of_search[drugs].count() / drugs_p_crimdam
criminal_damage_prc = df_clean.object_of_search[criminal_damage].count() / drugs_p_crimdam
# this will plot a pie chart for us
labels = 'Controlled Drugs', 'Articles for use in criminal damage'
fracs = [drugs_prc, criminal_damage_prc]
explode = (0,0)
plt.axis("equal")
plt.title('Percentage Drug vs. Weapons search as pct of Drugs plus Weapons')
plt.pie(fracs, explode=explode, labels=labels, autopct='%.0f%%', shadow=True);
df_clean.groupby('object_of_search')['gender'].value_counts().unstack(0).plot.bar(title='Gender by Object of Search', figsize=(15,10));
gender_male = df_clean.gender == 'Male'
gender_female = df_clean.gender == 'Female'
gender_other = df_clean.gender == 'Other'
gender_male_prc = df_clean.gender[gender_male].count() / len(df_clean)
gender_female_prc = df_clean.gender[gender_female].count() / len(df_clean)
gender_other_prc = df_clean.gender[gender_other].count() / len(df_clean)
# this will plot a pie chart for us
labels = 'Male', 'Female', 'Other'
fracs = [gender_male_prc, gender_female_prc, gender_other_prc]
explode = (0.0,0,0)
plt.axis("equal")
plt.title('Percentage of Stop and Search by Gender')
plt.pie(fracs, explode=explode, labels=labels, autopct='%.0f%%', shadow=True);
age_u10 = df_clean.age_range == 'under 10'
age_10 = df_clean.age_range == '10-17'
age_18 = df_clean.age_range == '18-24'
age_25 = df_clean.age_range == '25-34'
age_34 = df_clean.age_range == 'over 34'
age_u10_prc = df_clean.age_range[age_u10].count() / len(df_clean)
age_10_prc = df_clean.age_range[age_10].count() / len(df_clean)
age_18_prc = df_clean.age_range[age_18].count() / len(df_clean)
age_25_prc = df_clean.age_range[age_25].count() / len(df_clean)
age_34_prc = df_clean.age_range[age_34].count() / len(df_clean)
 # this will plot a pie chart for us
labels = 'Under 10', '10-17', '18-24', '25-34', '35+'
fracs = [age_u10_prc, age_10_prc, age_18_prc, age_25_prc, age_34_prc]
explode = (0,0,0,0,0)
plt.axis("equal")
plt.title('Percentage of Stop and Search by Age range')
plt.pie(fracs, explode=explode, labels=labels, autopct='%.0f%%', shadow=True);
df_clean.groupby('object_of_search')['age_range'].value_counts().unstack(0).plot.bar(title='Age range by Object of Search', figsize=(15,10));
gender_count = df_clean.groupby('date')
gender_count = gender_count.gender.apply(pd.value_counts).unstack(-1).fillna(0)


gender_count.plot(kind='line',figsize=(20,10), title='Stop and Search Count by Gender');
race_count = df_clean.groupby('date')
race_count = race_count.ethnicity.apply(pd.value_counts).unstack(-1).fillna(0)

race_count.plot(kind='line',figsize=(20,10), title='Stop and Search Count by Race');
object_count = df_clean.groupby('date')
object_count = object_count.object_of_search.apply(pd.value_counts).unstack(-1).fillna(0)

object_count.plot(kind='line',figsize=(20,10), title='Stop and Search Count by Object of Search');
age_count = df_clean.groupby('date')
age_count = age_count.age_range.apply(pd.value_counts).unstack(-1).fillna(0)
age_count.plot(kind='line', figsize=(20,10), title='Stop and Search Count by Age');
