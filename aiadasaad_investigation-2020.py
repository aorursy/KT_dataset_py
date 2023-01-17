 # First I import the packages that I will  use to facilitate the process of analysis the dataset

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# to Load dataset 

ds=pd.read_csv("../input/datasetcsv/Dataset.csv")

# to print the first few lines

ds.head()

#the niumber of colums and rows in the dataset

ds.shape
#To see if there are dublicated data or not

sum(ds.duplicated())
# To obtain a summary descriptive statstics about the dataset

ds.describe()
ds.info()
#Here clean the data errors

#First we rename the columns that have a wrong names

ds = ds.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap', 'No-show': 'No_Show' })

# deleting the non valid values of the age that have a value less than 0

ds=ds[ds.Age > 0]

# Convert the type of the AppointmentDay 

ds['AppointmentDay']=pd.to_datetime(ds["AppointmentDay"])

ds['ScheduledDay']=pd.to_datetime(ds["ScheduledDay"])

ds.info()
ds.drop(["PatientId","AppointmentID","ScheduledDay",'Neighbourhood'], axis=1, inplace=True)

ds.head()
ds.info()
ds.head()
fig, ax =plt.subplots(figsize=(10,7))

a= ds["No_Show"].value_counts()

print("The number od the patients whow show up is {} and te num of those who did not show up is {}".format(a[0],a[1]))

a.plot(kind="bar", title="No_Show vs Show");

ax.set_xlabel(" Show'No'    No_show'Yes' ")

ax.set_ylabel("Num of No_show and Show patients")
#This will give us the rows in which the patient show the mediacl appointment

Show=ds.No_Show=='No'

#This will give us the rows in which the patient no show the mediacl appointment

No_Show=ds.No_Show=='Yes'
#to see whether the Age affect showing up for the medical appointment or not 

fig, ax =plt.subplots(figsize=(10,7))

ds.Age[Show].hist(alpha=1,label='show');

ds.Age[No_Show].hist(alpha=1, label='no_show');

ax.set_title("Age Distripution")

ax.set_xlabel("Age value")

ax.set_ylabel("Show and No_Show")

plt.legend()
fig, ax =plt.subplots(figsize=(8,8))

#count the num of males and females

a=ds.Gender.value_counts()

#count the num of males and females who show up for the medial appointment

b=ds.Gender[Show].value_counts()

b.plot(kind='bar')

ax.set_title("Gender Distripution")

ax.set_xlabel("F for Female, M for Males")

ax.set_ylabel("Show and No_Show")

#count the num of males and females who did not  show up for the medial appointment

c=ds.Gender[No_Show].value_counts()

# the percentage of the female who show up 

f=(b[0]/a[0])*100

# the percentage of the female who did not show up 

f1=(c[0]/a[0])*100

# the percentage of the male who show up

m=(b[1]/a[1])*100

# the percentage of the male who did not show up



m1=(c[1]/a[1])*100

print("The percentage of the female who show up is {:0.2f} % and the percentage who did not show up is {:0.2f}% " .format(f,f1))

print("the percentage of the male who show up is {:0.2f} % and the percentage who did not show up is {:0.2f} %".format(m,m1))


def plotfunc(l):

    #l represnt a data of a column for example l can be ds.Scholarship

    #to see whether the factor affect showing up for the medical appointment or not 

    fig, ax =plt.subplots(figsize=(10,7))

    ax.hist( l[Show],alpha=0.5,label='show');

    ax.hist( l[No_Show],alpha=1, label='no_show');

    ax.set_title("Result of {}".format(l.name))

    ax.set_xlabel(l.name)

    ax.set_ylabel("Show and No_Show")

    plt.legend()



    #this is the num of the patient with and withput factor

    wo=l.value_counts()

    #this is the num of the patient with and withput factor and show up 

    s=(l[Show]).value_counts()

    #this is the num of the patient with and withput factor and did not show up 

    n=(l[No_Show]).value_counts()

    sw=(s[1]/wo[1])*100

    so=(s[0]/wo[0])*100

    name=l.name

    print("The percentage of patients with {} and show up is {:.2f} %".format(name,sw))

    print("The percentage of patients WithOUt {} and show up is {:.2f} %".format(name,so))  

# Here this function give us details about the scholarship

plotfunc(ds.Scholarship)
plotfunc(ds.Hypertension)
plotfunc(ds.Diabetes)
plotfunc(ds.Alcoholism)
plotfunc(ds.Handicap)
plotfunc(ds.SMS_received)
A=ds[Show]
#this function return the data of a certain condition to thoose patients who show up

def sdata( a,b, c):

    x=b==c

    d=a[x]

    return d
Show_Scholarship=sdata(A,A.Scholarship,1)

Show_Scholarship_No_Hypertension=sdata(Show_Scholarship,Show_Scholarship.Hypertension,0)

Show_S_No_Hyper_No_Alco=sdata(Show_Scholarship_No_Hypertension,Show_Scholarship_No_Hypertension.Alcoholism,0)

Show_no_Daibets=sdata(Show_S_No_Hyper_No_Alco,Show_S_No_Hyper_No_Alco.Diabetes,0)

show_no_Handicap=sdata(Show_no_Daibets,Show_no_Daibets.Handicap,0)

##This dataset represent the patients who show up and have no any condition health

Show_Scholarship_Without_Health_Conditions=show_no_Handicap
#this is the Total num of the patients who Show up with Scholarship and without health condition

z=Show_Scholarship_Without_Health_Conditions.No_Show.value_counts()

#this is the total num of the patients who show up 

q=A.No_Show.value_counts()

print("the percentage of patients who show up and have a scholarship but no any health conditions is {:0.2f}".format((z[0]/q[0])*100))
from subprocess import call

call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])