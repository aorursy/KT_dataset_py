# Importing packages planned to use for data analysis and predictive analysis.

import pandas as pd

import numpy as np

import statsmodels.api as sm

import matplotlib.pyplot as plt

%matplotlib inline
# Loading and printing out a few lines. Perform operations to inspect data

df=pd.read_csv('../input/noshowappointments-kagglev2-may-2016.csv')

df.head(3)
#Performing operations to inspect data types and look for instances of missing or possibly errant data.

df.info()
# Converting ScheduledDay and AppointmentDay to datetime format and further checking for clarity.

df['ScheduledDay']=pd.to_datetime(df['ScheduledDay'])

df['AppointmentDay']=pd.to_datetime(df['AppointmentDay'])

df.info()
# Renaming columns as per requirement

df=df.rename(columns ={'No-show':'No_show'})
# To get descriptive statistics for each column

df.describe()
# As it is clear from above data that the minimum age is negative which is not possible so dropping

# that partcular row

df.query('Age <0')
# Dropping that particular row

df.drop(df.index[99832],inplace=True)
# Checking the result

df.query('Age <0')
#Filtering the dataset on the basis of patients who did show up at the appointment and those who

# didn't, further dividing them into two different datasets.

df1=df.query('No_show == "No" ')

df2=df.query('No_show == "Yes" ')
df1.describe()
df2.describe()
# Adding new column and intialising it

df1.loc[:,'Age_group']=" "

df2.loc[:,'Age_group']=" ";
# Checking the results

df1.head(1)
# Filling the new column with values as defined above

w=df1['Age']

for i,c in enumerate(w):

    if (c<=18):

        df1.iloc[i,-1]='Minor'

    elif(c>18 and c<=30):

        df1.iloc[i,-1]='Adult'

    elif(c>30 and c<=60):

        df1.iloc[i,-1]='Mature'

    else:

        df1.iloc[i,-1]='Senior_Citizen'    
# Similarly for df2 dataframe

w1=df2['Age']

for i,c in enumerate(w1):

    if (c<=18):

        df2.iloc[i,-1]='Minor'

    elif(c>18 and c<=30):

        df2.iloc[i,-1]='Adult'

    elif(c>30 and c<=60):

        df2.iloc[i,-1]='Mature'

    else:

        df2.iloc[i,-1]='Senior_Citizen'
# Checking for results

df1.head(2)
# Similarly checking for df2

df2.head(2)
# Using pie-chart to answer the above question for both sections od dataset df1 and df2.

# For df1 the Pie-chart is

age_dist=df1['Age_group'].value_counts()

age_dist.plot(kind='pie',figsize=(20,10));
# Similarly for df2 dataset

age_dist1=df2['Age_group'].value_counts()

age_dist1.plot(kind='pie',figsize=(20,10));
# Here bar charts have been used to compare the proportions of students who received scholarships

# Comparing the proportions for those who did show up at the appointment

c1=df1['Scholarship'].value_counts()

k=["No","Yes"]

plt.bar(k,[c1[0]/(c1[0]+c1[1]),c1[1]/(c1[0]+c1[1])])

plt.title("Distribution of patients having received scholarships who did show up at the Appointment")

plt.xlabel("Scholarship status")

plt.ylabel("Number of Patients");
# Similarly comparing the proportions for those who did not show up at the appointment

c2=df2['Scholarship'].value_counts()

plt.bar(k,[c2[0]/(c2[0]+c2[1]),c2[1]/(c2[0]+c2[1])])

plt.title("Distribution of patients having received scholarships who did not show up at the Appointment")

plt.xlabel("Scholarship status")

plt.ylabel("Number of Patients");
# First dividing the the df2 dataset into two groups of males and females

df2_m=df2.query('Gender == "M"')

df2_f=df2.query('Gender == "F"')
# Now plotting the proportions of males and females who are suffereing from alcoholism but did not show up

# at the Appointment

count_m=df2_m['Alcoholism'].value_counts()

k1=["Males","Females"]

count_f=df2_f['Alcoholism'].value_counts()

plt.bar(k1,[count_m[1]/(count_m[1]+count_m[0]),count_f[1]/(count_f[0]+count_f[1])])

plt.title("Distribution of patients on gender who did not attend the appointment but are suffering from Alcoholism")

plt.xlabel("Gender")

plt.ylabel("Number of Patients");
# Creating a copy of above dataset

data=df.copy()
# Dropping columns

data.drop(columns=['PatientId','AppointmentID','ScheduledDay','AppointmentDay','Neighbourhood'],axis=1,inplace=True)
# Checking for results

data.head(2)
# Mapping the values for logistic regression

data['Gender']=data['Gender'].map({'F':0,'M':1})

data['No_show']=data['No_show'].map({'No':0,'Yes':1})
# Splitting the dataset in 9:1 ratio

train_data=data.iloc[:99474,:]

test_data=data.iloc[99474:,:]
estimators=['Gender','Age','Scholarship','Hipertension','Diabetes','Alcoholism','Handcap','SMS_received']



X = train_data[estimators]

y = train_data['No_show']
reg_log=sm.Logit(y,X)

result_log=reg_log.fit()

result_log.summary2()
def confusion_matrix(data,actual_values,model):

        

        # Confusion matrix 

        

        # Parameters

        # ----------

        # data: data frame or array

            # data is a data frame formatted in the same way as your input data (without the actual values)

            # e.g. const, var1, var2, etc. Order is very important!

        # actual_values: data frame or array

            # These are the actual values from the test_data

            # In the case of a logistic regression, it should be a single column with 0s and 1s

            

        # model: a LogitResults object

            # this is the variable where you have the fitted model 

            # e.g. results_log in this course

        # ----------

        

        #Predict the values using the Logit model

        pred_values = model.predict(data)

        # Specify the bins 

        bins=np.array([0,0.5,1])

        # Create a histogram, where if values are between 0 and 0.5 tell will be considered 0

        # if they are between 0.5 and 1, they will be considered 1

        cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]

        # Calculate the accuracy

        accuracy = (cm[0,0]+cm[1,1])/cm.sum()

        # Return the confusion matrix and 

        return cm, accuracy
# Checking the accuracy

confusion_matrix(X,y,result_log)
estimators=['Gender','Age','Scholarship','Hipertension','Diabetes','Alcoholism','Handcap','SMS_received']



X1 = test_data[estimators]

y1 = test_data['No_show']
# Checking the accuracy

confusion_matrix(X1,y1,result_log)
from subprocess import call

call(['python', '-m', 'nbconvert', 'Investigating_a_Dataset_Project.ipynb'])