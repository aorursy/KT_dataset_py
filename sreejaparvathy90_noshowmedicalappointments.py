# loading all the required libraries. Libraries for reading CSV File, Data Exploration, Visualization and Data Modeling 
# are all loaded in the cell.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.pyplot as plt 
import seaborn as sns
from datetime import datetime
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
noshow_appointments = pd.read_csv("../input/KaggleV2-May-2016.csv")
print (noshow_appointments.head())
# Assign the dataset to a variable and take a sample of the set.
df = pd.DataFrame(noshow_appointments)
df.head(10)
# Displays some necessary features.
# df.describe()
# Rename the columns with typos.
noshow_appointments.rename(columns = {'Hipertension':'Hypertension',
                                      'Handcap':'Handicap','No-show':'Noshow'},inplace=True)
print (noshow_appointments.columns)
noshow_appointments.head
df = pd.DataFrame(noshow_appointments)
#df.describe()
df.info()
# Converting AppointmentDay and ScheduledDay from object type to datetime64[ns].
df.AppointmentDay = df.AppointmentDay.apply(np.datetime64)
df['WeekDay'] = df['AppointmentDay'].dt.day
df.ScheduledDay = df.ScheduledDay.apply(np.datetime64)
df['DayScheduled'] = df['ScheduledDay'].dt.day
# Dropping 'PatientId' and 'AppointmentID' as they are just some system generated numbers which are not at all important 
# in our analysis.
df.drop(['PatientId', 'AppointmentID'], axis=1, inplace=True)
# Converting 'Gender' and 'Noshow' from object format to integer format.
df.Gender = df.Gender.apply(lambda x: 1 if x == 'M' else 0)
df['Noshow'] = df['Noshow'].replace('Yes',1)
df['Noshow'] = df['Noshow'].replace('No',0)
# Trying a sample dataset.
df.head(5)
#df.info()
# Print Unique Values
print("Unique Values in 'Gender' => {}".format(df.Gender.unique()))
print("Unique Values in 'Age' => {}".format(df.Age.unique()))
print("Unique Values in 'Scholarship' => {}".format(df.Scholarship.unique()))
print("Unique Values in 'Neighbourhood' => {}".format(df.Neighbourhood.unique()))
print("Unique Values in 'Hypertension' => {}".format(df.Hypertension.unique()))
print("Unique Values in 'Diabetes' => {}".format(df.Diabetes.unique()))
print("Unique Values in 'Alcoholism' => {}".format(df.Alcoholism.unique()))
print("Unique Values in 'Handicap' => {}".format(df.Handicap.unique()))
print("Unique Values in 'SMS_received' => {}".format(df.SMS_received.unique()))
print("Unique Values in 'WeekDay' => {}".format(df.WeekDay.unique()))
print("Unique Values in 'DayScheduled' => {}".format(df.DayScheduled.unique()))
print("Unique Values in 'Noshow' => {}".format(df.Noshow.unique()))
# Considering ages between 0 and 95 including both and drop the rest by considering them as outliers.
df.drop(df[(df.Age < 0) & (df.Age > 95)].index, inplace = True)
df.info()
# Age Distribution
df.plot(kind = "hist",y = "Age",bins = 100,range = (0,95)) 
plt.title('Age Distribution')
plt.xlabel('Age')
plt.show() 
# Analysing the distribution among genders with their age.
range_df = pd.DataFrame()
range_df['Age'] = range(95)
men = range_df.Age.apply(lambda x:len(df[(df.Age == x) & (df.Gender == 1)]))
women = range_df.Age.apply(lambda x:len(df[(df.Age == x) & (df.Gender == 0)]))
plt.plot(range(95), men, color = 'b')
plt.plot(range(95), women, color = 'r')
plt.legend([1,0])
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Gender based difference')
# Analysing and plotting the distribution among different medical conditions of both Genders with their age.
men_Hypertension = range_df[range_df.columns[0]].apply(lambda x: len(df[(df.Age == x) & (df.Gender == 1) & (df.Hypertension == 1)]))
women_Hypertension = range_df[range_df.columns[0]].apply(lambda x: len(df[(df.Age == x) & (df.Gender == 0) & (df.Hypertension == 1)]))

men_Diabetes = range_df[range_df.columns[0]].apply(lambda x: len(df[(df.Age == x) & (df.Gender == 1) & (df.Diabetes == 1)]))
women_Diabetes = range_df[range_df.columns[0]].apply(lambda x: len(df[(df.Age == x) & (df.Gender == 0) & (df.Diabetes == 1)]))

men_Alcoholism = range_df[range_df.columns[0]].apply(lambda x: len(df[(df.Age == x) & (df.Gender == 1) & (df.Alcoholism == 1)]))
women_Alcoholism = range_df[range_df.columns[0]].apply(lambda x: len(df[(df.Age == x) & (df.Gender == 0) & (df.Alcoholism == 1)]))

men_Handicap = range_df[range_df.columns[0]].apply(lambda x: len(df[(df.Age == x) & (df.Gender == 1) & (df.Handicap == 1)]))
women_Handicap = range_df[range_df.columns[0]].apply(lambda x: len(df[(df.Age == x) & (df.Gender == 0) & (df.Handicap == 1)]))

plt.figure(figsize = (10,10))
plt.subplot(2,2,1)
plt.plot(range(95),men_Hypertension/men)
plt.plot(range(95),women_Hypertension/women, color = 'r')
plt.title('Hypertension')
plt.legend([1,0], loc = 2)
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.subplot(2,2,2)
plt.plot(range(95),men_Diabetes/men)
plt.plot(range(95),women_Diabetes/women, color = 'r')
plt.title('Diabetes')
plt.legend([1,0], loc = 2)
plt.xlabel('Age')
plt.ylabel('Frequency')


plt.subplot(2,2,3)
plt.plot(range(95),men_Alcoholism/men)
plt.plot(range(95),women_Alcoholism/women, color = 'r')
plt.title('Alcoholism')
plt.legend([1,0], loc = 2)
plt.xlabel('Age')
plt.ylabel('Frequency')


plt.subplot(2,2,4)
plt.plot(range(95),men_Handicap/men)
plt.plot(range(95),women_Handicap/women, color = 'r')
plt.title('Handicap')
plt.legend([1,0], loc = 2)
plt.xlabel('Age')
plt.ylabel('Frequency')



app_day = sns.countplot(x = 'WeekDay', hue = 'Noshow', data = df)
app_day.set_title('AppointmentDay')
plt.show()
# Plotting SMS_received.
sms = sns.countplot(x = 'SMS_received',hue = 'Noshow',data = df)
sms.set_title('SMS_received')
# Plotting Scholarship received with people who are showing up or not.
schrsp = sns.countplot(x = 'Scholarship',hue = 'Noshow',data = df)
schrsp.set_title('Scholarship received')
# Plotting the distribution of Neighbourhood with pie-chart.
plt.subplots(figsize=(8,4))
df['Neighbourhood'].value_counts()[:10].plot(kind='pie',autopct='%1.1f%%',shadow=True,explode=[0.1,0,0,0,0,0,0,0,0,0])
plt.title('Distribution Of Top Neighbourhood')
plt.show()
# plotting ScheduledDay
datescheduled = sns.countplot(x = 'DayScheduled', hue = 'Noshow', data = df)
datescheduled.set_title('ScheduledDay')
df['Month Number'] = df['ScheduledDay'].dt.month
monthly_showup = sns.countplot(x = "Month Number", hue = "Noshow", data = df)
monthly_showup.set_title("Monthly Noshow Count")
plt.show()
#df.info()
df.head()
# Using sklearn Logistic Regression funtion to predict the accuracy of the No-show status.
# Get the Dependent and Independent Features.
x = df.drop(['Noshow','Neighbourhood','ScheduledDay','AppointmentDay'], axis=1)
y = df['Noshow']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)
acc_log
#Limitations.
# Conclusion according to the prediction.