import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pylab import *



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



dataset = pd.read_csv("../input/No-show-Issue-Comma-300k.csv")

dataset.tail()
# Check variables



# Age histogram



figure(figsize=(10,7))

dataset["Age"].hist(bins=116)

ylabel("count")

xlabel("age")

show()



# zoom on 0

figure(figsize=(10,7))

dataset["Age"].hist(bins=116,log=True)

ylabel("count")

xlabel("age")

xlim(-3,5)

show()



# there are a few negative values here 

print(dataset["Age"].min())

print((dataset["Age"]<0).sum())

print(dataset[dataset["Age"]<0])

# 6 negative age values



# Age histogram by gender

figure(figsize=(10,7))

dataset["Age"][dataset["Gender"]=='M'].hist(bins=116,alpha=0.5,color='b',range=(-2,113),label="men")

dataset["Age"][dataset["Gender"]=='F'].hist(bins=116,alpha=0.5,color='g',range=(-2,113),label="women")

legend()

ylabel("count")

xlabel("age")

show()
# Appointment day



print("Appointements on Monday :",(dataset["DayOfTheWeek"]=='Monday').sum())

print("Appointements on Tuesday :",(dataset["DayOfTheWeek"]=='Tuesday').sum())

print("Appointements on Wednesday :",(dataset["DayOfTheWeek"]=='Wednesday').sum())

print("Appointements on Thursday :",(dataset["DayOfTheWeek"]=='Thursday').sum())

print("Appointements on Friday :",(dataset["DayOfTheWeek"]=='Friday').sum())

print("Appointements on Saturday :",(dataset["DayOfTheWeek"]=='Saturday').sum())

print("Appointements on Sunday :",(dataset["DayOfTheWeek"]=='Sunday').sum())
# No show fraction 



print("No show fraction : ",float((dataset["Status"]=="No-Show").sum())/float(dataset.shape[0]))
# Fraction of the different conditions



print("Diabetes fraction :",float(dataset["Diabetes"].sum())/float(dataset.shape[0]))

print("Alcoolism fraction :",float(dataset["Alcoolism"].sum())/float(dataset.shape[0]))

print("HiperTension fraction :",float(dataset["HiperTension"].sum())/float(dataset.shape[0]))

print("Handcap fraction :",float(dataset["Handcap"].sum())/float(dataset.shape[0]))

print("Smokes fraction :",float(dataset["Smokes"].sum())/float(dataset.shape[0]))

print("Scholarship fraction :",float(dataset["Scholarship"].sum())/float(dataset.shape[0]))

print("Tuberculosis fraction :",float(dataset["Tuberculosis"].sum())/float(dataset.shape[0]))

print("Sms_Reminder fraction :",float(dataset["Sms_Reminder"].sum())/float(dataset.shape[0]))
# Awaiting time histogram



figure(figsize=(10,7))

dataset['AwaitingTime'].hist(bins=398,log=True)

ylabel("count")

xlabel("AwaitingTime")

show()
# Convert dates to two new feature : time at registration (as float), and number of days between registration and appointment (as int)



dataset["AppointmentRegistration"]=pd.to_datetime(dataset["AppointmentRegistration"])

dataset["ApointmentData"]=pd.to_datetime(dataset["ApointmentData"])



dataset["Registration_time"] = dataset["AppointmentRegistration"].dt.hour + dataset["AppointmentRegistration"].dt.minute/60. + dataset["AppointmentRegistration"].dt.second/60.



dataset['AppointmentRegistration'] = dataset['AppointmentRegistration'].apply(lambda x: x.date())

dataset['ApointmentData'] = dataset['ApointmentData'].apply(lambda x: x.date())



dataset['Time_interval'] = (dataset['ApointmentData']-dataset['AppointmentRegistration']).dt.days



print(dataset["Registration_time"])

print(dataset['Time_interval'])

# Convert all categorical variables to binary (one hot)

dataset['Male']=(dataset['Gender']=='M').astype('int')

dataset['Female']=(dataset['Gender']=='F').astype('int')

dataset['Monday']=(dataset['DayOfTheWeek']=='Monday').astype('int')

dataset['Tuesday']=(dataset['DayOfTheWeek']=='Tuesday').astype('int')

dataset['Wednesday']=(dataset['DayOfTheWeek']=='Wednesday').astype('int')

dataset['Thursday']=(dataset['DayOfTheWeek']=='Thursday').astype('int')

dataset['Friday']=(dataset['DayOfTheWeek']=='Friday').astype('int')

dataset['Saturday']=(dataset['DayOfTheWeek']=='Saturday').astype('int')

dataset['Sunday']=(dataset['DayOfTheWeek']=='Sunday').astype('int')

dataset['No-Show']=(dataset['Status']=="No-Show").astype('int')



dataset.describe()
print (dataset.corr())