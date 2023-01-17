import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv('../input/noshowappointments/KaggleV2-May-2016.csv')
df.head()
df.shape
df.describe()
# Checking the unique values for Handcap

df.Handcap.value_counts()
df.info()
print("Duplicated Values")
print(f"PatientId: {df.PatientId.duplicated().sum()}")
print(f"AppointmentId: {df.AppointmentID.duplicated().sum()}")
len(set(df.PatientId))
len(df.PatientId.value_counts())
print(f"There are {df.shape[0] - len(df.PatientId.value_counts())} patients that have scheduled more than one appointments. Meaning, the {round((df.shape[0]-len(df.PatientId.value_counts()))/df.shape[0]*100,2)}% of all the patients")
df.columns = map(str.lower, df.columns)

df.head(2)
np.sum(df.age == -1)
np.sum(df.handcap > 1)
df.head(2)
df.appointmentday = df.appointmentday.apply(lambda x:x[:10])
df.scheduledday = df.scheduledday.apply(lambda x:x[:10])
df.head()
df.appointmentday = pd.to_datetime(df.appointmentday)
df.scheduledday = pd.to_datetime(df.scheduledday)
df.info()
print("Looking at the min and max values of appointment day we can see our data contains: ")
print(f"First date of appointments: {min(df.appointmentday)}")
print(f"Last date of appointments: {max(df.appointmentday)}")
print(f"Total of {(max(df.appointmentday)-min(df.appointmentday))}")
print("Looking at the min and max values of schedule days we can see our data contains: ")
print(f"First date for scheduled appointments: {min(df.scheduledday)}")
print(f"Last date for scheduled appointments: {max(df.scheduledday)}")
print(f"Total of {(max(df.scheduledday)-min(df.scheduledday))}")
df.columns
df["waiting_time"] =df["appointmentday"] - df["scheduledday"] 
df.waiting_time = df.waiting_time.astype(str)
df.waiting_time = df.waiting_time.apply(lambda x: x.split(" ")[0])

df.tail()
df.waiting_time = df.waiting_time.astype(int)
df["waiting_time"].describe()
df.waiting_time[df.waiting_time<0].value_counts()
df['day_of_week'] = df['appointmentday'].dt.day_name()
df.head()
app_per_customer = dict()
number_of_appointment = dict()
for i in range(len(df.patientid)):
    if df.patientid[i] in app_per_customer.keys():
        app_per_customer[df.patientid[i]] += 1
        
    else:
        app_per_customer[df.patientid[i]] = 1
        
    number_of_appointment[i] = app_per_customer[df.patientid[i]]
    
apps_per_customer = pd.DataFrame.from_dict(number_of_appointment, orient='index', columns=["number_of_appointments"])
apps_per_customer.head()
df_2 = pd.concat([df, apps_per_customer], axis = 1,ignore_index=False)
df_2.head()
df_2 = df_2.query('age >= 0')
df_2 = df_2.query('handcap <= 1')
df_2.describe()
df_2.columns
df_3 = df_2.drop(['appointmentid', 'scheduledday', 'appointmentday'], axis=1)
df_3.head()
no_show_df = df_3[df_3["no-show"] == 'Yes']
show_df = df_3[df_3["no-show"] == 'No']


label = ["Showed", "Didn't showed"]
fig, ax = plt.subplots( figsize=(8,5))
height = len(show_df), len(no_show_df)
ax.bar(x=label, height=height)
ax.set_title("Relationship of the two categories of Appointments", fontsize=14)
ax.set_ylabel("Quantity", fontsize=20);
# Creating a dataframe with no duplicated clients. 
no_duplicates_df = df_3.drop_duplicates(subset='patientid', keep='first')
no_duplicates_df.tail()
scholarship_df_sum = pd.DataFrame(no_duplicates_df.groupby('no-show').sum()['scholarship'])
scholarship_df_mean = pd.DataFrame(no_duplicates_df.groupby('no-show').mean()['scholarship'])
print(scholarship_df_sum)
scholarship_df_sum.plot(kind='bar', rot=0, fontsize=15, legend = False)
plt.title("Quantity of scholarship depending on the show-up outcome")
plt.xlabel("Didn't show", fontsize=15)
plt.ylabel("Amount of scholarships", fontsize=15);
print(scholarship_df_mean)
scholarship_df_mean.plot(kind='bar', rot=0, fontsize=15, legend = False)
plt.title("Average of scholarship depending on the show-up outcome")
plt.xlabel("Didn't show", fontsize=15)
plt.ylabel("Average of scholarships", fontsize=15);
def plot_factors(var):

    var_df_mean = pd.DataFrame(no_duplicates_df.groupby('no-show').mean()[var])
    print(var_df_mean)
    var_df_mean.plot(kind='bar', rot=0, fontsize=15, legend = False)
    plt.title(f"Average of {var} depending on the show-up outcome")
    plt.xlabel("Didn't show", fontsize=15)
    plt.ylabel(f"Average of {var}", fontsize=15)

plot_factors('hipertension')
plot_factors('diabetes')
plot_factors('alcoholism')
plot_factors('handcap')
plot_factors('sms_received')
df_3.head()
weekday_df = df_3.groupby(['day_of_week', 'no-show']).count()['patientid']
# weekday_df = weekday_df.reset_index()
weekday_df.index
new_index = [(   'Monday',  'No'),
            (   'Monday', 'Yes'),
            (  'Tuesday',  'No'),
            (  'Tuesday', 'Yes'),
            ('Wednesday',  'No'),
            ('Wednesday', 'Yes'),
            ( 'Thursday',  'No'),
            ( 'Thursday', 'Yes'),           
            (   'Friday',  'No'),
            (   'Friday', 'Yes'),
            ( 'Saturday',  'No'),
            ( 'Saturday', 'Yes')]

weekday_df = weekday_df.reindex(new_index)
weekday_df = weekday_df.reset_index()
weekday_df
# Creating a new column
total_per_day = weekday_df.groupby('day_of_week').sum()['patientid']
percent = dict()
for i in range(len(weekday_df.day_of_week)):
    percent[i] = round((weekday_df['patientid'][i]/total_per_day[weekday_df['day_of_week'][i]])*100,2)
    
percent
percent_df = pd.DataFrame.from_dict(percent, orient='index', columns=['percent'])
weekday_df = pd.concat([weekday_df, percent_df], axis=1)
weekday_df
plt.figure(figsize=(8,5))
sns.barplot(x='day_of_week', y='percent', hue='no-show', data=weekday_df, saturation=0.55)
plt.title("Relative frequency of show ups per weekday", fontsize=14)
plt.xlabel("Weekdays", fontsize=14)
plt.ylabel("Percent of no-shown", fontsize=14);
round(df_3.waiting_time.mean())
over_waiting_time_df = df_3.query('waiting_time > 10')
less_waiting_time_df = df_3.query('waiting_time <= 10')
over_time_df = pd.DataFrame(over_waiting_time_df.groupby('no-show').count()['patientid'])
over_time_df.rename(columns = {'patientid':'over_waiting_time'}, inplace=True) 
total = over_time_df.over_waiting_time.sum()
over_time_df,total

# Relative frequency
over_time_df.over_waiting_time[0] = round((over_time_df.over_waiting_time[0]/total)*100,2)
over_time_df.over_waiting_time[1] = round((over_time_df.over_waiting_time[1]/total)*100,2)
over_time_df
less_time_df = pd.DataFrame(less_waiting_time_df.groupby('no-show').count()['patientid'])
less_time_df.rename(columns = {'patientid':'below_waiting_time'}, inplace=True) 
total = less_time_df.sum()
less_time_df, total
# Relative frequency
less_time_df.below_waiting_time[0] = round((less_time_df.below_waiting_time[0]/total)*100,2)
less_time_df.below_waiting_time[1] = round((less_time_df.below_waiting_time[1]/total)*100,2)
less_time_df
time_df = pd.concat([over_time_df, less_time_df], axis=1 )
time_df
time_df.reset_index()
plt.figure(figsize=(8,5))
barWidth = 0.25

plt.bar(x = barWidth, height=time_df.below_waiting_time[0], label='No', width=barWidth, color='maroon')
plt.bar(x = barWidth*2, height=time_df.below_waiting_time[1], label='Yes', width=barWidth, color='seagreen')

plt.bar(x = barWidth*5, height=time_df.over_waiting_time[0], label='No', width=barWidth, color='maroon')
plt.bar(x = barWidth*6, height=time_df.over_waiting_time[1], label='Yes', width=barWidth, color='seagreen')

#plt.bar(x = barWidth+barWidth, height=[time_df.below_waiting_time[1],time_df.over_waiting_time[1]], label='Yes', width=barWidth)
# sns.barplot(x=['low_waiting_time', 'high_waiting_time'], y=['over_waiting_time', 'below_waiting_time'], hue='no-show', data=time_df, saturation=0.55)
plt.title("Relative frequency of show ups according the waiting time", fontsize=14)
plt.xlabel("Waiting Time", fontsize=14)
plt.ylabel("Percent of no-shown", fontsize=14)

plt.xticks([barWidth*1.5,  barWidth*5.5], 
           ['Below average ', 'Over average']) 
plt.legend( title = "Didn't show up?",labels=('No', 'Yes'), fontsize='medium');
df_3.columns
one_app_df = df_3.query('number_of_appointments == 1')
more_app_df = df_3.query('number_of_appointments > 1 & number_of_appointments < 5')
many_more_app_df = df_3.query('number_of_appointments >= 5 & number_of_appointments < 10')
lot_more_app_df = df_3.query('number_of_appointments >= 10')
one_app_df_2 = pd.DataFrame(one_app_df.groupby('no-show').count()['patientid'])
one_app_df_2.rename(columns={'patientid': 'one_appointment'}, inplace=True)
total = one_app_df_2.one_appointment.sum()
one_app_df_2, total
one_app_df_2["one_appointment"][0] = round(one_app_df_2["one_appointment"][0]/total*100,2)
one_app_df_2["one_appointment"][1] = round(one_app_df_2["one_appointment"][1]/total*100,2)
one_app_df_2
more_app_df_2 = pd.DataFrame(more_app_df.groupby('no-show').count()['patientid'])
more_app_df_2.rename(columns={'patientid': 'between_1_and_5'}, inplace=True)
total = more_app_df_2.between_1_and_5.sum()
more_app_df_2, total
more_app_df_2["between_1_and_5"][0] = round(more_app_df_2["between_1_and_5"][0]/total*100,2)
more_app_df_2["between_1_and_5"][1] = round(more_app_df_2["between_1_and_5"][1]/total*100,2)
more_app_df_2
many_more_app_df_2 = pd.DataFrame(many_more_app_df.groupby('no-show').count()['patientid'])
many_more_app_df_2.rename(columns={'patientid': 'between_5_and_10'}, inplace=True)
total = many_more_app_df_2.between_5_and_10.sum()
many_more_app_df_2, total
many_more_app_df_2["between_5_and_10"][0] = round(many_more_app_df_2["between_5_and_10"][0]/total*100,2)
many_more_app_df_2["between_5_and_10"][1] = round(many_more_app_df_2["between_5_and_10"][1]/total*100,2)
many_more_app_df_2
lot_more_app_df_2 = pd.DataFrame(lot_more_app_df.groupby('no-show').count()['patientid'])
lot_more_app_df_2.rename(columns={'patientid': 'more_than_10'}, inplace=True)
total = lot_more_app_df_2.more_than_10.sum()
lot_more_app_df_2
lot_more_app_df_2["more_than_10"][0] = round(lot_more_app_df_2["more_than_10"][0]/total*100,2)
lot_more_app_df_2["more_than_10"][1] = round(lot_more_app_df_2["more_than_10"][1]/total*100,2)
lot_more_app_df_2
number_app_df = pd.concat([one_app_df_2, more_app_df_2, many_more_app_df_2, lot_more_app_df_2], axis=1)
number_app_df
plt.figure(figsize=(8,5))
barWidth = 0.25

columns = ['one_appointment', 'between_1_and_5', 'between_5_and_10', 'more_than_10']
index = [1,5,9,13]
for i,col in zip(index,columns):
    
    plt.bar(x = barWidth*i, height=number_app_df[col][0], label='No', width=barWidth, color='maroon')
    plt.bar(x = barWidth*(i+1), height=number_app_df[col][1], label='Yes', width=barWidth, color='seagreen')
    
plt.xlabel("Number of appointments", fontsize=14)
plt.ylabel("Percent of show up", fontsize=14)

plt.xticks([barWidth*1.5,  barWidth*5.5, barWidth*10, barWidth*13.5], 
           ['Only 1 ', 'Between 1 and 5 ', 'Between 5 and 10 ', 'Over 10']) 
plt.legend( title = "Didn't show up?",labels=('No', 'Yes'), fontsize='medium');
one_app_df.neighbourhood.describe()
df_neigh = pd.DataFrame(one_app_df["neighbourhood"].value_counts())
df_neigh.head(), df_neigh.max()

df_neigh.describe()


df_neigh = df_neigh.query('neighbourhood > 216')
len(df_neigh)
index = df_neigh.index
one_app_df_2 = pd.DataFrame(one_app_df.query("neighbourhood in @index"))
one_app_df_2.tail()
one_app_df_2["no-show"].value_counts()
one_app_df_3 = one_app_df_2[one_app_df_2['no-show'] == 'Yes']
one_app_df_3.head()
df_grouped = pd.DataFrame(one_app_df_3.groupby("neighbourhood").count()["patientid"])
df_sorted_neigh = df_grouped.sort_values(by="patientid", ascending=False)
df_sorted_neigh.head()

df_sorted_neigh.rename(columns={"patientid": "non_showed_appointments"}, inplace=True)

len(df_sorted_neigh)
df_app = pd.DataFrame(one_app_df_2.groupby("neighbourhood").count()["patientid"])
# df_text = df_text.query("Neighbourhood in @y_1")
df_app.rename(columns= {"patientid":"number_of_appointments"}, inplace=True)
len(df_app)
df_final = pd.concat([df_sorted_neigh,df_app], axis=1, join='inner')
df_final.head()
len(df_final)
df_final["percent_of_nonshown_appointments"] = round(df_final['non_showed_appointments']/df_final['number_of_appointments'],2)*100
df_final.head()
df_final.sort_values(by="percent_of_nonshown_appointments", ascending=False, inplace=True)
df_final.head()
width=df_final["percent_of_nonshown_appointments"][:20]
width = width[::-1]
y = y_1 = df_final.head(20).index
y_1 = y_1[::-1]

fig, ax = plt.subplots(figsize=(10,8))

ax.barh(y=y, width=width)
plt.title('Top 20 neigbourhood with no show-up appointment', fontsize=14)
plt.xlabel('Relative value Non show-up appointment/total of appointments', fontsize=14)
plt.ylabel('Neighbourhood', fontsize=16)
for i, v in enumerate(width):
    ax.text(v + 0.05, i , str(int(v))+'%', color='black', fontweight='bold')
