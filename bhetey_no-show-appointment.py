# This Python 3 environment comes with many helpful analytics libraries installed
# import modules needed for the investigation 
%matplotlib inline
import numpy as np 
import pandas as pd
from datetime import datetime as dt 
import matplotlib.pyplot as plt

# import the data from the csv and assigned it to a variable called df 
# using the header shows us the first 5 roles from top of the data set.
# Here, the data is imported into the system and read using the pd. 
df = pd.read_csv('../input/KaggleV2-May-2016.csv')
df = df.dropna(axis=1)
df.head()
df.shape   # this gives us the idea of the total columns we have in the data set.
# Here i am renaming some columns to provide some clarity into the data like removing spaces
df.rename(columns = { "PatientId":'Patient_Id', 'Hipertension': 'Hypertension', \
                     "AppointmentID":'Appointment_ID',"ScheduledDay": "Scheduled_Day", \
                     'AppointmentDay': 'Appointment_Day' }, inplace=True)
df.columns = df.columns.str.replace('-', '_')
df.columns
df.info() # this is showing more information about the data set like the type of data they are
# Here we converted the date, time and seconds using pandas in built function to make the date and time more
# comprehensive to understand
# below is the link for the documentation 
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html#pandas.to_datetime
scheduled_day = pd.to_datetime(df['Scheduled_Day'])
appointment_day = pd.to_datetime(df['Appointment_Day'])
schedule_appointment = ['Scheduled_Day', 'Appointment_Day']
schedule_appointment = df[schedule_appointment]
schedule_appointment.head(5)
teens_start_age = 18 
teenagers = df[df.Age < teens_start_age]    # no of people below the age of 18
print ("Here is the number of people below the age of 18 : " + str(len(teenagers)))
settled_down_age = 40   # to ask my mentor if i have to turn this into a function.
settling_down = df[(df['Age'] >= teens_start_age) & (df['Age'] <= settled_down_age)]
len(settling_down)
print ("This is the number of people from 18 to 40 of the data set : " + str(len(settling_down)))
old_age = 60    # Here i looked at the age over 40 
getting_old = df[(df['Age'] > settled_down_age) & (df['Age'] <= old_age)]
len(getting_old)
print ("This is the number of people over 40 till 60 years : " + str(len(getting_old)))
finally_old = df[df.Age > old_age]    # Here i want to know the people above the age of 60 
len(finally_old)
print("Here is the nummber of people over 60 years : " + str(len(finally_old)))

female_in_the_data = df[df.Gender == "F"]    # no of females in the gender columns
male_in_the_data = df[df.Gender == "M"]    # no of males in the gender columns 
len(male_in_the_data)
len(female_in_the_data)
print (" Here is the number of Males : " + str(len(male_in_the_data)))
print (" Here is the number of Females : "+ str(len(female_in_the_data)))
age_mean = df['Age'].mean() # mean of the age distribution 
age_standard_deviation = df['Age'].std() # standard deviation of the age distribution 
print ('This is the mean : ',age_mean)
print ('This is the standard deviation : ',age_standard_deviation)
%matplotlib inline
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

mu = age_mean  # mean of distribution
sigma = age_standard_deviation  # standard deviation of distribution

num_bins = 50

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(df['Age'], num_bins, normed=1)

# add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)
ax.plot(bins, y, '--')
ax.set_xlabel('Age of the people')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of People Age')
#$\mu=, $\sigma=$
# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()
male_no_shows = df[(df['Gender'] == "M") & (df['No_show'] == 'Yes')]
males_no_shows = int(len(male_no_shows))
print ("Male that did not show up for appointment : ",males_no_shows)
# Here i am trying to get the female in the gender that did not show for the appointment
female_no_shows = df[(df['Gender'] == 'F') & (df['No_show'] == 'Yes')]
females_no_shows = int(len(female_no_shows))
print ("Female that did not show up for appointment : ",females_no_shows)
# Here, we are looking at the number of males that showed up 
male_shows_up = df[(df['Gender'] == "M") & (df['No_show'] == 'No')]
males_shows_up = int(len(male_shows_up))
print (" Males that showed up for the appointment : ", males_shows_up)
#We are taking a look at the number of females that showed up for the appointment 
female_shows_up = df[(df['Gender'] == 'F') & (df['No_show'] == 'No')]
females_shows_up = int(len(female_shows_up))
print (" Females that showed up for the Medical appointment: ",females_shows_up)
grouping_gender_noshow = df.groupby(['Gender','No_show'])
grouping_gender_noshow.size().unstack().plot(kind='pie', subplots = True, 
                                            autopct='%1.1f%%', figsize=(15,15)
                                            ,title = 'Pie showing the relating of Gender with No Show')
grouping_gender_noshow.size().unstack().plot(kind='bar', figsize=(15,15), title = 'Bar chart of No show  grouped by Gender')
from __future__ import division
percentage = 100
total_number_shows = len(df)
total_number_shows = int(total_number_shows)
def calculating_percentage(input_show):
    the_percentage = ((input_show / total_number_shows)*percentage)
    print (the_percentage,'%')

male_not_showed = calculating_percentage(males_no_shows)
female_not_showed = calculating_percentage(females_no_shows)
male_that_showed = calculating_percentage(males_shows_up)
female_that_showed = calculating_percentage(females_shows_up)
df['Day_of_the_appointment'] = df['Appointment_Day'].dt.weekday_name    # the day of the week was formatted here

Monday_Appointment = df[(df['Day_of_the_appointment'] == 'Monday')]
len(Monday_Appointment)
def checking_for_days(appointment_day, day_of_week):
    appointment = df[(appointment_day == day_of_week)]
    print (day_of_week+'s : ' + str(len(appointment)))
    return len(appointment)
    
max_of_days = [
    checking_for_days(df['Day_of_the_appointment'], 'Monday'),
    checking_for_days(df['Day_of_the_appointment'], 'Tuesday'),
    checking_for_days(df['Day_of_the_appointment'], 'Wednesday'),
    checking_for_days(df['Day_of_the_appointment'], 'Thursday'),
    checking_for_days(df['Day_of_the_appointment'], 'Friday'),
    checking_for_days(df['Day_of_the_appointment'], 'Saturday'),
    checking_for_days(df['Day_of_the_appointment'], 'Sunday')
]
# returning the number of appointment missed based on the weekday
print ('Here is the most days where appointment was missed : ', max(max_of_days))
# People missed most appointment on a Wednesday
df['Day_of_the_appointment'].mode()
%matplotlib inline
df['Day_of_the_appointment'].value_counts().plot(kind="pie", shadow = True, startangle=270, autopct='%1.1f%%', figsize=(10,10),
                                                 title = ('Percentage of days of the week'), legend = True)
ax = df['Day_of_the_appointment'].hist()
ax.set_xlabel("(Day of the week)")
ax.set_ylabel('Frequency')
