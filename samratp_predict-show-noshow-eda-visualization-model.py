import pandas as pd

import numpy as np

import datetime

from time import strftime



from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns
week_key = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df = pd.read_csv('../input/KaggleV2-May-2016.csv')
print("The shape of the DataFrame is => {}".format(df.shape))
df.info()
# Print the top 5 rows

df.head()
# Convert PatientId from Float to Integer

df['PatientId'] = df['PatientId'].astype('int64')



# Convert ScheduledDay and AppointmentDay from 'object' type to 'datetime64[ns]'

df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.date.astype('datetime64[ns]')

df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.date.astype('datetime64[ns]')



# Rename incorrect column names.

df = df.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap', 'SMS_received': 'SMSReceived', 'No-show': 'NoShow'})
df.info()
df.head()
print("Features in the DataFrame => {}".format(df.columns.ravel()))
# Drop 'PatientId' and 'AppointmentID' as they are just some system genrated numbers.

df.drop(['PatientId', 'AppointmentID'], axis=1, inplace=True)
# Print Unique Values

print("Unique Values in `Gender` => {}".format(df.Gender.unique()))

print("Unique Values in `Scholarship` => {}".format(df.Scholarship.unique()))

print("Unique Values in `Hypertension` => {}".format(df.Hypertension.unique()))

print("Unique Values in `Diabetes` => {}".format(df.Diabetes.unique()))

print("Unique Values in `Alcoholism` => {}".format(df.Alcoholism.unique()))

print("Unique Values in `Handicap` => {}".format(df.Handicap.unique()))

print("Unique Values in `SMSReceived` => {}".format(df.SMSReceived.unique()))
df['Scholarship'] = df['Scholarship'].astype('object')

df['Hypertension'] = df['Hypertension'].astype('object')

df['Diabetes'] = df['Diabetes'].astype('object')

df['Alcoholism'] = df['Alcoholism'].astype('object')

df['Handicap'] = df['Handicap'].astype('object')

df['SMSReceived'] = df['SMSReceived'].astype('object')
df.info()
# Print some sample data

df.sample(n=5)
# Print Unique Values for 'Age'

print("Unique Values in `Age` => {}".format(np.sort(df.Age.unique())))
print("Patients with `Age` less than -1 -> {}".format(df[df.Age == -1].shape[0]))

print("Patients with `Age` equal to 0 -> {}".format(df[df.Age == 0].shape[0]))
df = df[df.Age >= 0]
df[(df.Age <= 0) & ((df.Hypertension.astype(int) == 1) | (df.Diabetes.astype(int) == 1) | (df.Alcoholism.astype(int) == 1))]
# Print Unique Values for 'ScheduledDay'

print("Unique Values in `ScheduledDay` => {}".format(np.sort(df.ScheduledDay.dt.strftime('%Y-%m-%d').unique())))
# Print Unique Values for 'AppointmentDay'

print("Unique Values in `AppointmentDay` => {}".format(np.sort(df.AppointmentDay.dt.strftime('%Y-%m-%d').unique())))
# Print Unique Values for 'Neighbourhood'

print("Unique Values in `Neighbourhood` => {}".format(np.sort(df.Neighbourhood.unique())))
# Print Total Count for 'Neighbourhood'

print("Total Count for `Neighbourhood` => {}".format(df.Neighbourhood.unique().size))
# Get Day of the Week for ScheduledDay and AppointmentDay

df['ScheduledDay_DOW'] = df['ScheduledDay'].dt.weekday_name

df['AppointmentDay_DOW'] = df['AppointmentDay'].dt.weekday_name
df['AppointmentDay'] = np.where((df['AppointmentDay'] - df['ScheduledDay']).dt.days < 0, df['ScheduledDay'], df['AppointmentDay'])



# Get the Waiting Time in Days of the Patients.

df['Waiting_Time_days'] = df['AppointmentDay'] - df['ScheduledDay']

df['Waiting_Time_days'] = df['Waiting_Time_days'].dt.days
# Sanity check to see if the Waiting Time is less than Zero for any of the data points.

print("There are [{}] records where the Waiting Time is less than Zero.".format(df[df.Waiting_Time_days < 0].shape[0]))
df.info()
df.sample(n=10)
print("NoShow and Show Count of Patients\n")

print(df.groupby(['NoShow']).size())



print("\nNoShow and Show '%' of Patients\n")

show = df.groupby(['NoShow']).size()[0]/(df.groupby(['NoShow']).size()[0]+df.groupby(['NoShow']).size()[1])

print("Percent of Patients who `Showed Up` => {:.2f}%".format(show*100))

noshow = df.groupby(['NoShow']).size()[1]/(df.groupby(['NoShow']).size()[0]+df.groupby(['NoShow']).size()[1])

print("Percent of Patients who Did `Not Showed Up` => {:.2f}%".format(noshow*100))
ax = sns.countplot(x=df.NoShow, data=df)

ax.set_title("Show/NoShow Patients")

plt.show()
ax = sns.countplot(x=df.Gender, hue=df.NoShow, data=df)

ax.set_title("Show/NoShow for Females and Males")

x_ticks_labels=['Female', 'Male']

ax.set_xticklabels(x_ticks_labels)

plt.show()
plt.figure(figsize=(16,2))

plt.xticks(rotation=90)

_ = sns.boxplot(x=df.Age)
plt.figure(figsize=(16,4))

plt.xticks(rotation=90)

ax = sns.countplot(x=df.Age)

ax.set_title("No of Appointments by Age")

plt.show()
plt.figure(figsize=(16,4))

plt.xticks(rotation=90)

ax = sns.countplot(x=df.Age, hue=df.NoShow)

ax.set_title("Show/NoShow of Appointments by Age")

plt.show()
df_age_ratio = df[df.NoShow == 'No'].groupby(['Age']).size()/df.groupby(['Age']).size()
plt.figure(figsize=(16,4))

plt.xticks(rotation=90)

ax = sns.barplot(x=df_age_ratio.index, y=df_age_ratio)

ax.set_title("Percentage of Patients that Showed Up by Age")

plt.show()
plt.figure(figsize=(16,4))

plt.xticks(rotation=90)

ax = plt.hist(df_age_ratio)

plt.title("Distribution of Percentage Show Up by Age")

plt.show()
plt.figure(figsize=(16,4))

plt.xticks(rotation=90)

ax = sns.countplot(x=np.sort(df.Neighbourhood))

ax.set_title("No of Appointments by Neighbourhood")

plt.show()
plt.figure(figsize=(16,4))

plt.xticks(rotation=90)

ax = sns.countplot(x=np.sort(df.Neighbourhood), hue=df.NoShow, order=df.Neighbourhood.value_counts().index)

ax.set_title("Show/NoShow by Neighbourhood")

plt.show()
df_n_ratio = df[df.NoShow == 'No'].groupby(['Neighbourhood']).size()/df.groupby(['Neighbourhood']).size()
plt.figure(figsize=(16,4))

plt.xticks(rotation=90)

ax = sns.barplot(x=df_n_ratio.index, y=df_n_ratio)

ax.set_title("Percetage Show Up of Patients by Neighbourhood")

plt.show()
ax = sns.countplot(x=df.Scholarship, hue=df.NoShow, data=df)

ax.set_title("Show/NoShow for Scholarship")

x_ticks_labels=['No Scholarship', 'Scholarship']

ax.set_xticklabels(x_ticks_labels)

plt.show()
df_s_ratio = df[df.NoShow == 'No'].groupby(['Scholarship']).size()/df.groupby(['Scholarship']).size()

ax = sns.barplot(x=df_s_ratio.index, y=df_s_ratio, palette="RdBu_r")

ax.set_title("Show Percentage for Scholarship")

x_ticks_labels=['No Scholarship', 'Scholarship']

ax.set_xticklabels(x_ticks_labels)

plt.show()
ax = sns.countplot(x=df.Hypertension, hue=df.NoShow, data=df)

ax.set_title("Show/NoShow for Hypertension")

x_ticks_labels=['No Hypertension', 'Hypertension']

ax.set_xticklabels(x_ticks_labels)

plt.show()
df_h_ratio = df[df.NoShow == 'No'].groupby(['Hypertension']).size()/df.groupby(['Hypertension']).size()

ax = sns.barplot(x=df_h_ratio.index, y=df_h_ratio, palette="RdBu_r")

ax.set_title("Show Percentage for Hypertension")

x_ticks_labels=['No Hypertension', 'Hypertension']

ax.set_xticklabels(x_ticks_labels)

plt.show()
ax = sns.countplot(x=df.Diabetes, hue=df.NoShow, data=df)

ax.set_title("Show/NoShow for Diabetes")

x_ticks_labels=['No Diabetes', 'Diabetes']

ax.set_xticklabels(x_ticks_labels)

plt.show()
df_d_ratio = df[df.NoShow == 'No'].groupby(['Diabetes']).size()/df.groupby(['Diabetes']).size()

ax = sns.barplot(x=df_d_ratio.index, y=df_d_ratio, palette="RdBu_r")

ax.set_title("Show Percentage for Diabetes")

x_ticks_labels=['No Diabetes', 'Diabetes']

ax.set_xticklabels(x_ticks_labels)

plt.show()
ax = sns.countplot(x=df.Alcoholism, hue=df.NoShow, data=df)

ax.set_title("Show/NoShow for Alcoholism")

x_ticks_labels=['No Alcoholism', 'Alcoholism']

ax.set_xticklabels(x_ticks_labels)

plt.show()
df_a_ratio = df[df.NoShow == 'No'].groupby(['Alcoholism']).size()/df.groupby(['Alcoholism']).size()

ax = sns.barplot(x=df_a_ratio.index, y=df_a_ratio, palette="RdBu_r")

ax.set_title("Show Percentage for Alcoholism")

x_ticks_labels=['No Alcoholism', 'Alcoholism']

ax.set_xticklabels(x_ticks_labels)

plt.show()
ax = sns.countplot(x=df.Handicap, hue=df.NoShow, data=df)

ax.set_title("Show/NoShow for Handicap")

plt.show()
df_ha_ratio = df[df.NoShow == 'No'].groupby(['Handicap']).size()/df.groupby(['Handicap']).size()

ax = sns.barplot(x=df_ha_ratio.index, y=df_ha_ratio, palette="RdBu_r")

ax.set_title("Show Percentage for Handicap")

plt.show()
ax = sns.countplot(x=df.SMSReceived, hue=df.NoShow, data=df)

ax.set_title("Show/NoShow for SMSReceived")

x_ticks_labels=['No SMSReceived', 'SMSReceived']

ax.set_xticklabels(x_ticks_labels)

plt.show()
df_s_ratio = df[df.NoShow == 'No'].groupby(['SMSReceived']).size()/df.groupby(['SMSReceived']).size()

ax = sns.barplot(x=df_s_ratio.index, y=df_s_ratio, palette="RdBu_r")

ax.set_title("Show Percentage for SMSReceived")

x_ticks_labels=['No SMSReceived', 'SMSReceived']

ax.set_xticklabels(x_ticks_labels)

plt.show()
plt.figure(figsize=(16,4))

ax = sns.countplot(x=df.ScheduledDay_DOW, order=week_key)

ax.set_title("Appointment Count for Scheduled Day of Week")

plt.show()
plt.figure(figsize=(16,4))

ax = sns.countplot(x=df.AppointmentDay_DOW, order=week_key)

ax.set_title("Appointment Count for Appointment Day of Week")

plt.show()
plt.figure(figsize=(16,4))

ax = sns.countplot(x=df.AppointmentDay_DOW, hue=df.NoShow, order=week_key)

ax.set_title("Show/NoShow for Appointment Day of the Week")

plt.show()
df_a_dow_ratio = df[df.NoShow == 'No'].groupby(['AppointmentDay_DOW']).size()/df.groupby(['AppointmentDay_DOW']).size()

plt.figure(figsize=(16,4))

ax = sns.barplot(x=df_a_dow_ratio.index, y=df_a_dow_ratio, order=week_key, palette="RdBu_r")

ax.set_title("Show Percent for Appointment Day of the Week")

plt.show()
plt.figure(figsize=(16,4))

ax = sns.countplot(x=df.Waiting_Time_days, order=df.Waiting_Time_days.value_counts().iloc[:55].index)

ax.set_title("Waiting Time in Days (Descending Order)")

plt.show()
plt.figure(figsize=(16,4))

ax = sns.countplot(x=df.Waiting_Time_days, order=df.Waiting_Time_days.value_counts(ascending=True).iloc[:55].index)

ax.set_title("Waiting Time in Days (Ascending Order)")

plt.show()
plt.figure(figsize=(16,4))

ax = sns.countplot(x=df.Waiting_Time_days, hue=df.NoShow, order=df.Waiting_Time_days.value_counts().iloc[:25].index)

ax.set_title("Show/NoShow Count for Waiting Time in Days (High Count)")

plt.show()
plt.figure(figsize=(16,4))

ax = sns.countplot(x=df.Waiting_Time_days, hue=df.NoShow, order=df.Waiting_Time_days.value_counts(ascending=True).iloc[:55].index)

ax.set_title("Show/NoShow Count for Waiting Time in Days (Low Count)")

plt.show()
plt.figure(figsize=(16,4))

ax = sns.countplot(x=df[['Waiting_Time_days']].sort_values('Waiting_Time_days', ascending=False).Waiting_Time_days.iloc[:400])

ax.set_title("Descending Waiting Time in Days")

plt.show()
plt.figure(figsize=(16,4))

ax = sns.countplot(x=df[['Waiting_Time_days']].sort_values('Waiting_Time_days', ascending=False).Waiting_Time_days.iloc[:400], hue=df.NoShow)

ax.set_title("Show/NoShow - Descending Waiting Time in Days")

plt.show()
df_w_ratio = df[df.NoShow == 'No'].groupby(['Waiting_Time_days']).size()/df.groupby(['Waiting_Time_days']).size()



plt.figure(figsize=(16,4))

ax = sns.barplot(x=df_w_ratio.index, y=df_w_ratio, order=df.Waiting_Time_days.iloc[:70].index, palette="BuGn_d")

ax.set_title("Percent of Show for Waiting Time in Days")

plt.show()
df.info()
# Use `LabelEncoder` to encode labels with value between 0 and n_classes-1.

#Gender

le = LabelEncoder()

df['Gender'] = le.fit_transform(df['Gender'])

#Neighbourhood

le = LabelEncoder()

df['Neighbourhood'] = le.fit_transform(df['Neighbourhood'])

#ScheduledDay_DOW

le = LabelEncoder()

df['ScheduledDay_DOW'] = le.fit_transform(df['ScheduledDay_DOW'])

#AppointmentDay_DOW

le = LabelEncoder()

df['AppointmentDay_DOW'] = le.fit_transform(df['AppointmentDay_DOW'])

print("LabelEncoder Completed")



#NoShow

le = LabelEncoder()

df['NoShow'] = le.fit_transform(df['NoShow'])
df['ScheduledDay_Y'] = df['ScheduledDay'].dt.year

df['ScheduledDay_M'] = df['ScheduledDay'].dt.month

df['ScheduledDay_D'] = df['ScheduledDay'].dt.day

df.drop(['ScheduledDay'], axis=1, inplace=True)



df['AppointmentDay_Y'] = df['AppointmentDay'].dt.year

df['AppointmentDay_M'] = df['AppointmentDay'].dt.month

df['AppointmentDay_D'] = df['AppointmentDay'].dt.day

df.drop(['AppointmentDay'], axis=1, inplace=True)
df.sample(n=10)
# Get the Dependent and Independent Features.

X = df.drop(['NoShow'], axis=1)

y = df['NoShow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
dt_clf = DecisionTreeClassifier(random_state=0)

dt_clf.fit(X_train, y_train)
print("Feature Importance:\n")

for name, importance in zip(X.columns, np.sort(dt_clf.feature_importances_)[::-1]):

    print("{} -- {:.2f}".format(name, importance))
dt_clf.score(X_test, y_test)
rf_clf = RandomForestClassifier(random_state=0)

rf_clf.fit(X_train, y_train)
print("Feature Importance:\n")

for name, importance in zip(X.columns, np.sort(rf_clf.feature_importances_)[::-1]):

    print("{} -- {:.2f}".format(name, importance))
rf_clf.score(X_test, y_test)
params={'n_estimators':[10,20], 'max_depth':[None, 5], 'min_samples_split':[2,3]}

rf_clf = RandomForestClassifier(random_state=0)

clf_grid = GridSearchCV(rf_clf, params, cv=5, n_jobs=-1, verbose=1)

clf_grid.fit(X, y)

print(clf_grid.best_params_)

print(clf_grid.best_score_)