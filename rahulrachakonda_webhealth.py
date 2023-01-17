import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
# Load data to a dataframe

df = pd.read_csv('../input/KaggleV2-May-2016.csv')



# Learn the size of the dataset

df.shape


df.columns
# Typos in the column names as well as their format should be corrected / unified

df.columns = ['patient_id', 'appointment_id', 'gender', 'scheduled_day', 

              'appointment_day', 'age', 'neighbourhood', 'scholarship', 'hypertension',

              'diabetes', 'alcoholism', 'handicap', 'sms_received', 'no_show']

df.columns


df.head(5)
# Check how many patients_ids are not integers

non_int_patient_ids = df[~ df.patient_id.apply(lambda x: x.is_integer())]

print('There are {} patients_ids that are not integers'.format(len(non_int_patient_ids)))

non_int_patient_ids
# Extract float patient_ids from the list above

patient_ids = [93779.52927, 537615.28476, 141724.16655, 39217.84439, 43741.75652]

    

# Convert all float patient_ids to int (by truncating the decimal part)

# and check if such patients exist in the rest of the dataset

for i in range(len(patient_ids)):

    patient_ids[i] = int(patient_ids[i])

    if df.query('patient_id == {}'.format(patient_ids[i])).empty:

        print('Patient id == {} does not exist.'.format(patient_ids[i]))

    else:

        print('Patient id == {} already exists.'.format(patient_ids[i]))
# Convert patient_id from float to int

df['patient_id'] = df['patient_id'].astype('int64')



# Check if the patient_id is int64

df.info()
# Convert columns types

df['scheduled_day'] = pd.to_datetime(df['scheduled_day']).dt.date.astype('datetime64[ns]')

df['appointment_day'] = pd.to_datetime(df['appointment_day']).dt.date.astype('datetime64[ns]')



# Check if the type is now datetime

df.info()
# Create awaiting_time_days column

df['awaiting_time_days'] = (df.appointment_day - df.scheduled_day).dt.days # and convert timedelta to int



# Check if the column exists

df.info()
# Create appointment_dow column

df['appointment_dow'] = df.scheduled_day.dt.weekday_name



# Check the values

df['appointment_dow'].value_counts()
df.describe()
df.hist(figsize=(16,14));
# Print Unique Values

print("Unique Values in `gender` => {}".format(df.gender.unique()))
# Print Unique Values

print("Unique Values in `scheduled_day` => {}".format(df.scheduled_day.unique()))
# Print Unique Values

print("Unique Values in `appointment_day` => {}".format(df.appointment_day.unique()))
# Print Unique Values

print("Unique Values in `age` => {}".format(df.age.unique()))
print('Before change')

print("Patients with `Age` less than -1 -> {}".format(df[df.age == -1].shape[0]))

print("Patients with `Age` equal to 0 -> {}".format(df[df.age == 0].shape[0]))

print("Patients with `Age` greater than 110 -> {}".format(df[df.age > 110].shape[0]))



df = df[(df.age >= 0) & (df.age <= 110)]

df.age.value_counts()



print('After change')

print("Patients with `Age` less than -1 -> {}".format(df[df.age == -1].shape[0]))

print("Patients with `Age` equal to 0 -> {}".format(df[df.age == 0].shape[0]))

print("Patients with `Age` greater than 110 -> {}".format(df[df.age > 110].shape[0]))
# Let's see a boxplot showing what is age values distribution (already seen above in a histogram and basic descriptive statistics table)

plt.figure(figsize=(16,2))

plt.xticks(rotation=90)

_ = sns.boxplot(x=df.age)
# Let's see how many there are patients of each age

plt.figure(figsize=(16,4))

plt.xticks(rotation=90)

ax = sns.countplot(x=df.age)

ax.set_title("No of appointments by age")

plt.show()
# Print Unique Values

print("Unique Values in `scholarship` => {}".format(df.scholarship.unique()))
# Print Unique Values

print("Unique Values in `hypertension` => {}".format(df.hypertension.unique()))
# Print Unique Values

print("Unique Values in `diabetes` => {}".format(df.diabetes.unique()))
# Print Unique Values

print("Unique Values in `alcoholism` => {}".format(df.alcoholism.unique()))


# Print Unique Values

print("Unique Values in `handicap` => {}".format(df.handicap.unique()))
# The handicap column contains 4 numeric values (classes), which is unusual comparing to other cathegorical variables in the dataset

df.handicap.value_counts()
# Print Unique Values

print("Unique Values in `sms_received` => {}".format(df.sms_received.unique()))
# Print Unique Values

print("Unique Values in `awaiting_time_days` => {}".format(df.awaiting_time_days.unique()))


print('Before change: {}'.format(df[(df.awaiting_time_days < 0)].awaiting_time_days.value_counts()))





df = df[(df.awaiting_time_days >= 0)]



print('After change: {}'.format(df[(df.awaiting_time_days < 0)].awaiting_time_days.value_counts()))
plt.figure(figsize=(16,4))

plt.xticks(rotation=90)

ax = sns.countplot(x=df.awaiting_time_days)

ax.set_title("No of patients by awaiting time in days")

plt.show()


awaiting0 = df[(df.awaiting_time_days == 0)].awaiting_time_days.value_counts()

awaiting0
awaiting0_not_showed_up = len(df.query('awaiting_time_days  == 0 and no_show == "Yes"'))

awaiting0_not_showed_up_ratio = int(round(awaiting0_not_showed_up/awaiting0[0]*100))

print('Out of all patients scheduling an appointment for the same day (in total {}), {} of patients did not show up ({}%).'.format(awaiting0[0], 

                                                                                                                                   awaiting0_not_showed_up, 

                                                                                                                                   awaiting0_not_showed_up_ratio))
# It seems that most of the visits happened within 3 months from being scheduled

sns.stripplot(data = df, y = 'awaiting_time_days', jitter = True)

plt.ylim(0, 200)

plt.show();


print('Scheduling visits started on: {}.'.format(df['scheduled_day'].min()))

print('Scheduling visits ended on: {}.'.format(df['scheduled_day'].max()))



fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('scheduled_day')

ax.set_ylabel('frequency')

df['scheduled_day'].hist();
print('Visit appointments started on: {}.'.format(df['appointment_day'].min()))

print('Visit appointments ended on: {}.'.format(df['appointment_day'].max()))



fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(1,1,1)

ax.set_xlabel('scheduled_day')

ax.set_ylabel('frequency')

df['appointment_day'].hist(grid=False, ax=ax);
# Print Unique Values

print("Unique Values in `appointment_dow` => {}".format(df.appointment_dow.unique()))
# Are the appointments ids unique?

# If yes, then num_unique_apps will be equal to number of all records in our dataset

num_unique_apps = len(df.appointment_id.unique())

all_dataset_rec_number = df.shape[0]

print('{} == {}'.format(num_unique_apps, all_dataset_rec_number))
all_appointments = df.shape[0]

missed_appointments = len(df.query('no_show == \'Yes\''))

missed_ratio = int(round(missed_appointments/all_appointments*100))



ax = sns.countplot(x=df.no_show, data=df)

ax.set_title("Show / No-Show Patients")

plt.show();



print('{}% of appointments were missed.'.format(missed_ratio))
all_appointments_by_f = len(df.loc[df['gender'] == "F"])

all_appointments_by_m = len(df.loc[df['gender'] == "M"])



missed_appointments_by_f = len(df.query('no_show == "Yes" and gender == "F"'))

missed_appointments_by_m = len(df.loc[(df['gender'] == "M") & (df['no_show'] == "Yes")])



missed_ratio_f = int(round(missed_appointments_by_f/all_appointments_by_f*100))

missed_ratio_m = int(round(missed_appointments_by_m/all_appointments_by_m*100))



ax = sns.countplot(x=df.gender, hue=df.no_show, data=df)

ax.set_title("Show / No-Show for Females and Males")

x_ticks_labels=['Female', 'Male']

plt.show();



print('Out of {} appointments made by females, {} were missed with the ratio of {}%.'.format(all_appointments_by_f, missed_appointments_by_f, missed_ratio_f))

print('Out of {} appointments made by males, {} were missed with the ratio of {}%.'.format(all_appointments_by_m, missed_appointments_by_m, missed_ratio_m))
df.patient_id.value_counts().iloc[0:10]

# First, let's look at categorical variables

categorical_vars = ['gender', 'scholarship', 'hypertension', 'diabetes', 'alcoholism', 'handicap', 'sms_received', 'appointment_dow']



fig = plt.figure(figsize=(16, 11))

for i, var in enumerate(categorical_vars):

    ax = fig.add_subplot(3, 3, i+1)

    df.groupby([var, 'no_show'])[var].count().unstack('no_show').plot(ax=ax, kind='bar', stacked=True)
# Two useful masks to be used in further analysis

showed = df.no_show == 'No'

not_showed = df.no_show == 'Yes'
# Let's now look closer to numerical variables

# Age:

df.age[showed].hist(alpha=0.8, bins=20);

df.age[not_showed].hist(alpha=0.8, bins=20);
# Number of days between the date of scheduling an appointment and the appointment itself

df.awaiting_time_days[showed].hist(alpha=0.8, bins=20);

df.awaiting_time_days[not_showed].hist(alpha=0.8, bins=20);
# This is a helper column representing no_shows in a numerical form (Yes->1, No->0)

df['no_show_numeric'] = np.where(df['no_show']=='Yes', 1, 0)
df[['diabetes', 'no_show_numeric']].groupby(['diabetes'], as_index=False).mean().sort_values(by='no_show_numeric', ascending=False)
grid = sns.FacetGrid(df, col='no_show_numeric', row='diabetes', height=4.4, aspect=1.6)

grid.map(plt.hist, 'age', alpha=.5, bins=20)

grid.add_legend();
# The Pointplot uses bootstraping method to estimate of a mean and a std error

# Appointments related to patients with diabetes and receiving SMS:

grid = sns.FacetGrid(df, row='sms_received', height=4.4, aspect=1.6)

grid.map(sns.pointplot, 'diabetes', 'no_show_numeric', 'gender', palette='deep', dodge=True)

grid.add_legend();
# Appointments related to patients with hypertension and receiving SMS:

grid = sns.FacetGrid(df, row='sms_received', height=4.4, aspect=1.6)

grid.map(sns.pointplot, 'hypertension', 'no_show_numeric', 'gender', palette='deep', dodge=True)

grid.add_legend();
# Appointments related to patients with diabetes and participation in scholarship:

grid = sns.FacetGrid(df, row='scholarship', height=4.4, aspect=1.6)

grid.map(sns.pointplot, 'diabetes', 'no_show_numeric', 'gender', palette='deep', dodge=True)

grid.add_legend();
# Appointments related to patients with hypertension and participation in scholarship:

grid = sns.FacetGrid(df, row='scholarship', height=4.4, aspect=1.6)

grid.map(sns.pointplot, 'hypertension', 'no_show_numeric', 'gender', palette='deep', dodge=True)

grid.add_legend();