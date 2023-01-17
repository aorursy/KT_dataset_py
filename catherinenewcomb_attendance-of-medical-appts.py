import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



df = pd.read_csv("../input/attendance-of-medical-appointments-in-brazil/noshowappointments-kagglev2-may-2016.csv")
df.rename(columns= {"PatientId": "patient_id", "AppointmentID": "appointment_id", "Gender": "gender", "Age": "age", "Neighbourhood": "neighborhood", "Scholarship": "scholarship", "Hipertension": "hypertension", "Diabetes": "diabetes", "Alcoholism": "alcoholism", "Handcap": "handicap", "SMS_received": "SMS_received", "No-show": "no_show", "ScheduledDay" : "scheduled_day", "AppointmentDay" : "appointment_day", }, inplace=True)
df.info()
df["scheduled_day"] = pd.to_datetime(df["scheduled_day"])

df["appointment_day"] = pd.to_datetime(df["appointment_day"],yearfirst=True)

df["time_diff"] = df["appointment_day"] - df["scheduled_day"]

df.head()
plt.style.use("seaborn-talk")



ax = df["time_diff"].astype('timedelta64[D]').plot.hist(title="Turnaround for Scheduled Appointments", figsize=(12, 5));

ax.set_xlabel("Days");
df.time_diff.describe()
t = [-1, 1, 3, 14, 179]

t_bins = pd.to_timedelta(t, unit='days')

labels = ["same day", "1 to 3 days", "4 to 14 days", "More than 14 days"]



df['time_diff_bins'] = pd.cut(df['time_diff'], bins=t_bins, labels=labels)

df.groupby("time_diff_bins").no_show.value_counts()
df["time_diff"] = df["time_diff"].astype('timedelta64[D]')

df.drop(df.index[(df.time_diff <= -2.0)], inplace=True)
df["handicap"].unique()

df["handicap"].value_counts()

df["any_handicap"] = pd.cut(df["handicap"], bins=[-1, 0, 4], labels=["No Handicap", "Handicap"])

df["any_handicap"].value_counts()
df["age"].value_counts()

print(df.query("age > 100"))
sum(df.duplicated())

print(len(df["appointment_id"].unique()))

print(len(df["patient_id"].unique()))

print(len(df["patient_id"]))

print(110522 - 62299)
print(df["no_show"].value_counts())

no_shows = 22314

print((no_shows / df.shape[0])*100)
#Boolstring describes the condition you want a percentage of and characteristic is a string describing it

def no_show_percentage(bool_string, characteristic): 

    query1_total = len(df.query(bool_string + "and no_show == 'Yes'" ))

    query2_total = len(df.query(bool_string))

    return_pct = (query1_total / query2_total) * 100

    print("Percentage of patients {} who were no-show:".format(characteristic), return_pct)



no_show_percentage("scholarship == 1", "receiving scholarship" )

no_show_percentage("scholarship == 0", "receiving no scholarship" )
no_show_percentage("any_handicap == 'Handicap'", "with handicaps")

no_show_percentage("any_handicap == 'No Handicap'", "with no handicaps")
no_show_percentage("hypertension == 1", "with hypertension")

no_show_percentage("hypertension == 0", "without hypertension")
no_show_percentage("diabetes == 1", "with diabetes")

no_show_percentage("diabetes == 0", "without diabetes")
df["age_range"] = pd.cut(df["age"], bins=[-2, 12, 25, 35, 45, 55, 65, 75, 115], 

                    labels= ["Under 12", "13 to 25", "26 to 35", "36 to 45", "46 to 55", 

                              "56 to 65", "66 to 75", "76 and up"])

def return_no_show_pct(bool_string): 

    query1_total = len(df.query(bool_string + "and no_show == 'Yes'" ))

    query2_total = len(df.query(bool_string))

    return_pct = (query1_total / query2_total) * 100

    return return_pct



pct_under_12 = return_no_show_pct("age_range == 'Under 12'")

pct_13_25 = return_no_show_pct("age_range == '13 to 25'")

pct_26_35 = return_no_show_pct("age_range == '26 to 35'")

pct_36_45 = return_no_show_pct("age_range == '36 to 45'")

pct_46_55 = return_no_show_pct("age_range == '46 to 55'")

pct_56_65 = return_no_show_pct("age_range == '56 to 65'")

pct_66_75 = return_no_show_pct("age_range == '66 to 75'")

pct_76_up = return_no_show_pct("age_range == '76 and up'")









age_percentages = [pct_under_12, pct_13_25, pct_26_35, pct_36_45, pct_46_55, 

                    pct_56_65, pct_66_75, pct_76_up]

labels = ("Under 12", "13 to 25", "26 to 35", "36 to 45", "46 to 55", 

                              "56 to 65", "66 to 75", "76 and up")

y_pos = np.arange(len(labels))



plt.style.use("seaborn-talk")





plt.figure(figsize=(12, 4))

plt.bar(y_pos, age_percentages, color='b');

plt.xticks(y_pos, labels);

plt.ylabel("Percentage of No Shows");

plt.xlabel("Age Group");

plt.title("Percentage of No Shows by Age Group");

ages_13_25 = df[(df["age"] > 13) & (df["age"] <= 25)]
no_show_percentage("gender == 'F'", "female")

no_show_percentage("gender == 'M'", "male")
r = '#b35806'

p = '#542788'

plt.figure(figsize=(9,4))

sns.countplot(data=df, x='time_diff_bins', hue='no_show', palette=[r, p]);

plt.legend(["Attended", "No-show"])

plt.xlabel("Turnaround for Scheduled Appointments");

plt.ylabel("Number of Appointments")

plt.title("Distribution of No-Shows vs. Attended by Time Turnaround");
#convert 'yes'/'no' in no_show column to integers to calculate percentages



df['no_show_num'] = 0

df.loc[df['no_show'] == 'Yes', 'no_show_num'] = 1

df.loc[df['no_show'] == 'No', 'no_show_num'] = 0

df['no_show_num'] = df['no_show_num'].astype(int)
no_show_by_turnaround = pd.DataFrame(df.groupby("time_diff_bins").no_show_num.mean())
plt.figure(figsize=(9,4))

sns.barplot(data=no_show_by_turnaround, x=no_show_by_turnaround.index, y='no_show_num',

            color='b')

plt.title("Percentage of No-Shows by Appointment Turnaround Time")

plt.xlabel("Number of Days Away from Time Appointment was Scheduled")

plt.title("Proportion of No Shows");


sns.countplot(data=df, x='SMS_received', hue='no_show', palette=[p, r])

plt.legend(["Attended", "No Show"])

plt.xticks([0, 1], ["No SMS Reminder", "SMS Reminder"])

plt.xlabel('');

plt.title("Appointment Attendance by Text Message Reminder");
sms_rec = df[df["SMS_received"] == 1]

no_sms = df[df["SMS_received"] == 0]

sms_rec = sms_rec.no_show_num.mean() * 100

no_sms = no_sms.no_show_num.mean() * 100

print(sms_rec, no_sms)
plt.figure(figsize=(5,4))



plt.bar([0,1], sms_rec, color=r)

plt.bar(0, 100 - sms_rec, bottom=sms_rec, color=p)

plt.bar(1, no_sms, color = r)

plt.bar(1, 100 - no_sms, bottom=no_sms, color=p)

plt.xticks([0, 1], ["SMS reminder", "No SMS reminder"])

plt.legend(["No Show", "Attended"])

plt.title("Appointment Attendance by Text Message Reminder")

plt.text(0, 23.2, '{:0.1f}%'.format(sms_rec), color='white', fontsize=12)

plt.text(1, 12.1, '{:0.1f}%'.format(no_sms), color='white', fontsize=12);
print("Patients who received a SMS reminder AND had a same day ppt:", 

              len(df.query("SMS_received == 1 and time_diff_bins == 'same day'")))

print("Patients who received a SMS reminder:", len(df.query("SMS_received == 1")))

print(df.groupby("SMS_received").time_diff_bins.value_counts())
SMS_received_adjusted = return_no_show_pct("SMS_received == 1 & time_diff_bins != 'same day'")

no_SMS_received_adjusted = return_no_show_pct("SMS_received == 0 & time_diff_bins != 'same day'")



plt.style.use("seaborn-talk")



plt.figure(figsize=(5,6))



plt.bar([0,1], SMS_received_adjusted, color=r)

plt.bar(0, 100 - SMS_received_adjusted, bottom=SMS_received_adjusted, color=p)

plt.bar(1, no_SMS_received_adjusted, color = r)

plt.bar(1, 100 - no_SMS_received_adjusted, bottom=no_SMS_received_adjusted, color=p)

plt.xticks([0, 1], ["SMS reminder", "No SMS reminder"])

plt.legend(["No Show", "Attended"])

plt.title("Appointment Attendance by Text Message Reminder \n Excluding Same Day Appts")

plt.text(0, 23.2, '{:0.1f}%'.format(SMS_received_adjusted), color='white', fontsize=12)

plt.text(1, 26.3, '{:0.1f}%'.format(no_SMS_received_adjusted), color='white', fontsize=12);