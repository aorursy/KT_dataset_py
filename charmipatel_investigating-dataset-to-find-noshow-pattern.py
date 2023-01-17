# Import required libraaries and data from csv file
import pandas as pd
import numpy as np
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('../input/KaggleV2-May-2016.csv')
df.head()
df.isnull().sum()
# None of the column contents null values

df.info()
df.describe()
df.shape
sum(df.duplicated())

gender = df['Gender'].value_counts()
gender
noshow = df['No-show'].value_counts()
noshow
df.drop(df[df['Age']<0].index,inplace=True)

df.describe()
#Check unique values for below columns

#Renaming columns with correct spelling
df.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap'}, inplace=True)

print(df['Scholarship'].unique())

print(df['Hypertension'].unique())
print(df['Diabetes'].unique())

print(df['Alcoholism'].unique())

print(df['Handicap'].unique())

print(df['SMS_received'].unique())
# Convert data type
df['PatientId'] = df['PatientId'].astype('int64')
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

df.info()

df['Age'].describe()
# Set  edges that will be used to divide the data into age groups
edges = [0, 15, 30, 45, 60, 75, 90, 115]
# Create labels for each age group
names = ['<15', '15-29', '30-44', '45-59', '60-74', '75-89', '>=90']

df['AgeGroup'] = pd.cut(df['Age'], edges, labels=names, right=False, include_lowest=True)
df.head()

age_group = df.groupby(['AgeGroup', 'No-show']).count()['Age']
age_group
below_fifteen = age_group['<15', 'No'] / df['AgeGroup'].value_counts()['<15']
fifteen_to_twenty_nine = age_group['15-29', 'No'] / df['AgeGroup'].value_counts()['15-29']
thirty_to_fourty_four = age_group['30-44', 'No'] / df['AgeGroup'].value_counts()['30-44']
fourty_five_to_fifty_nine = age_group['45-59', 'No'] / df['AgeGroup'].value_counts()['45-59']
sixty_and_seventy_four = age_group['60-74', 'No'] / df['AgeGroup'].value_counts()['60-74']
seventy_five_to_ninety = age_group['75-89', 'No'] / df['AgeGroup'].value_counts()['75-89']
more_than_ninety = age_group['>=90', 'No'] / df['AgeGroup'].value_counts()['>=90']
# Plot the graph 
proportions = [below_fifteen, fifteen_to_twenty_nine, thirty_to_fourty_four, fourty_five_to_fifty_nine, sixty_and_seventy_four, seventy_five_to_ninety, more_than_ninety]
plt.bar([1, 2, 3, 4, 5,6,7], proportions, width=0.3)
plt.xlabel('Age Group')
plt.ylabel('Proportion')
plt.xticks([1, 2, 3, 4, 5,6,7], ['<15', '15-29', '30-44', '45-59', '60-74', '75-90', '>=90'])
plt.title('Patients that showed up according to Age Groups');


# No-show between males and females
sns.set(style="darkgrid")
sns.countplot(x='No-show',hue='Gender',data=df)
plt.xlabel('Appointment_show according to gender')

# Lets find percentage to get clear idea
# We can say that, females have more no-show than man do. 
male_counts_with_noshow=df[(df['Gender']=='M') & (df['No-show']=='Yes')].count()
female_counts_with_noshow=df[(df['Gender']=='F') & (df['No-show']=='Yes')].count()
total_Men=len((df['Gender']=='M'))
total_Women=len((df['Gender']=='F'))
percentage_of_Women=(female_counts_with_noshow/total_Women)*100
percentage_of_men=(male_counts_with_noshow/total_Men)*100
print("Percentage of female who did not show up ",np.round(percentage_of_Women['Gender'],0),"%")
print("Percentage of male who did not show up ",np.round(percentage_of_men['Gender'],0),"%")

#Pie chart will give us more clear picture  
labels='Female','Male'
sizes=[13,7]
plt.pie(sizes,  labels=labels, autopct='%1.2f%%',
        shadow=True, startangle=90)
plt.title("Percent of males and females")

scholarship_counts = df.groupby(['Scholarship', 'No-show']).count()['Age']
scholarship_counts

no_scholarship = scholarship_counts[0, 'No'] / df['Scholarship'].value_counts()[0]

with_scholarship = scholarship_counts[1, 'No'] / df['Scholarship'].value_counts()[1]

no_scholarship, with_scholarship

hypertension_counts = df.groupby(['Hypertension', 'No-show']).count()['Age']
hypertension_yes = hypertension_counts[1, 'No'] / df['Hypertension'].value_counts()[1]
hypertension_no = hypertension_counts[0, 'No'] / df['Hypertension'].value_counts()[0]

diabetes_counts = df.groupby(['Diabetes', 'No-show']).count()['Age']
diabetic = diabetes_counts[1, 'No'] / df['Diabetes'].value_counts()[1]
non_diabetic = diabetes_counts[0, 'No'] / df['Diabetes'].value_counts()[0]

alcoholism_counts = df.groupby(['Alcoholism', 'No-show']).count()['Age']
alcoholic = alcoholism_counts[1, 'No'] / df['Alcoholism'].value_counts()[1]
non_alcoholic = alcoholism_counts[0, 'No'] / df['Alcoholism'].value_counts()[0]

index = np.array([1, 2, 3])
width = 0.25
plt.bar(index, [hypertension_no, non_diabetic, non_alcoholic], width=width, color='orange', label='Without the medical problem')
plt.bar(index+width, [hypertension_yes, diabetic, alcoholic], width=width, color='blue', label='With the medical problem')

plt.xlabel('Conditions')
plt.ylabel('Proportion that showed up')

locations = index + width / 2
plt.xticks(locations, ['Hypertension', 'Diabetes', 'Alcoholism'])
plt.title('Proportion of Patients that Showed Up, According to medical reasons')

plt.legend();

sms_counts = df.groupby(['SMS_received', 'No-show']).count()['Age']
no_sms_show_up = sms_counts[0, 'No'] / df['SMS_received'].value_counts()[0]
sms_show_up = sms_counts[1, 'No'] / df['SMS_received'].value_counts()[1]
plt.bar([1, 2], [no_sms_show_up, sms_show_up], width=0.6, color=['teal'])
plt.xlabel('SMS received')
plt.ylabel('Proportion')
plt.xticks([1, 2], ['No', 'Yes'])
plt.title('Proportion of Patients that Showed Up: SMS vs Without SMS');

df['day_week_appoitntment'] = df['AppointmentDay'].dt.weekday_name

weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
for index,i in enumerate(weekdays):
    j=df[df.day_week_appoitntment==i]
    count=len(j)
    total_count=len(df)

    perc=(count/total_count)*100
    print(i,count)
    plt.bar(index,perc)
    
plt.xticks(range(len(weekdays)),weekdays, rotation=45)
plt.title('Appointments as per day of the week')
plt.ylabel("Percent of appointment")
plt.show()

# No show day of the week
no_Show_Yes=df[df['No-show']=='Yes']
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
for index,i in enumerate(weekdays):
    k=no_Show_Yes[no_Show_Yes.day_week_appoitntment==i]
    count=len(k)
    total_count=len(no_Show_Yes)
    perc=(count/total_count)*100
    print(i,count,perc)
    plt.bar(index,perc)

plt.xticks(range(len(weekdays)),weekdays, rotation=60)
plt.xlabel("Days of week")
plt.ylabel("Percent ")
plt.title('Percent of No-Show per weekday')
plt.show()

no_Show_No=df[df['No-show']=='No']
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
for index,i in enumerate(weekdays):
    k=no_Show_No[no_Show_No.day_week_appoitntment==i]
    count=len(k)
    total_count=len(no_Show_No)
    perc=(count/total_count)*100
    print(i,count,perc)
    plt.bar(index,perc)

plt.xticks(range(len(weekdays)),weekdays, rotation=45)
plt.xlabel("Days of week")
plt.ylabel("Percent ")
plt.title('Percent of Show up per weekday')
plt.show()
