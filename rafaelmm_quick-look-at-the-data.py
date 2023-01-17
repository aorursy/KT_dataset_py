import numpy as np 
import pandas as pd 
import seaborn as sns

# Input data files are available in the "../input/" directory.

df_gender_age_train = pd.read_csv('../input/gender_age_train.csv')
sns.distplot(df_gender_age_train.age)
print("The average age in the train sample is: %.1f years" % df_gender_age_train.age.mean())
sns.set_palette("GnBu_d")

sns.countplot(x='group', data=df_gender_age_train, order = ['F23-', 'F24-26', 'F27-28', 'F29-32', 'F33-42', 'F43+'])
sns.countplot(x='group', data=df_gender_age_train, order = ['M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+'])
gender_counts = df_gender_age_train.gender.value_counts()
print("There are %d men and %d women in the Train set." % (gender_counts['M'], gender_counts['F']))
print("%d" % df_gender_age_train.device_id.duplicated().sum())
print("This means there are no repeated device_ids in the test set.")
df_events = pd.read_csv('../input/events.csv')

dates_and_times = pd.to_datetime(df_events.timestamp, format='%Y-%m-%d %H:%M:%S')
sns.countplot(dates_and_times.dt.day, order=[30, 1, 2, 3, 4, 5, 6, 7])
print("The events in the train set are contained within Abr/30 to Mar/07, with an similar distribution between Mar/01 and Mar/07.")
sns.countplot(dates_and_times.dt.hour)
sns.countplot(dates_and_times.dt.minute)
sns.countplot(dates_and_times.dt.second)
