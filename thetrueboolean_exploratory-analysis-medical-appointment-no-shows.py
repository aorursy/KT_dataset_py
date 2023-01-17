# Import necessary libraries and packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# load dataset to a pandas dataframe

df = pd.read_csv('../input/noshowappointments/KaggleV2-May-2016.csv')
# view top of dataset

df.head()
df.sample(5)
# view number of rows and columns

print(f'Dataset has {df.shape[0]} rows and {df.shape[1]} columns')
df.describe()
# basic stats of the data

df.describe()
# general info

df.info()
# view column data types

df.dtypes
# Check for duplicate rows

df.duplicated().sum()
df.PatientId.duplicated().sum()
df.AppointmentID.duplicated().sum()
df.query('Age < 0')
# view number of unique entries per column

df.nunique()
# # Let's see what these unique values in each column look like.



for col in df.iloc[:, np.r_[2,5,7:14]].columns:

    print(f"{col}   ===>  ", sorted(df[f'{col}'].unique()), '\n')


labels = ['patient_id', 'appointment_id', 'gender', 'scheduled_day', 'appointment_day','age',

          'neighbourhood', 'scholarship', 'hypertension','diabetes', 'alcoholism',

          'handicap', 'sms_received', 'status']

df.columns = labels
df.columns
df.scheduled_day = pd.to_datetime(df.scheduled_day)

df.appointment_day = pd.to_datetime(df.appointment_day)
df['patient_id'] = df['patient_id'].astype('int64')

df['patient_id'] = df['patient_id'].astype('str')
df['gender'] = df['gender'].astype('category')

df['sms_received'] = df['sms_received'].astype('category')

df['hypertension'] = df['hypertension'].astype('category')

df['status'] = df.status.astype('category')
df.status.cat.rename_categories({'No':'Show','Yes':'No Show'}, inplace = True)



#similarly

df.gender.cat.rename_categories({'F':'Female','M':'Male'}, inplace = True)

df.hypertension.cat.rename_categories({0:'Not Hypertensive', 1:'Hypertensive'}, inplace = True)

df.sms_received.cat.rename_categories({0:'Not Received', 1:'Received'}, inplace = True)
df.dtypes
df.drop(df.query('age == -1').index, axis=0, inplace=True)
## To confirm that worked

df.query('age == -1')
df['waiting_days'] = (df.appointment_day - df.scheduled_day).abs().dt.days

df.waiting_days.head(8)
df['age_range'] = pd.cut(df.age, labels = ['0 - 4','5 - 14','15 - 24', '25 - 44',

                            '45 - 64', '65 - 115'], bins = [-1,5,15,25,45,65,116])



df['wait_category'] = pd.cut(df.waiting_days, bins=[-1,0,7,30,90,180], include_lowest=True,

                         labels=['no_wait','1_week','1_month','3_months','6_months'])

df_copy = df.copy()

df_copy.head()
for i,j in enumerate(df.columns):

    print(i,j, end='\t')
df = df.iloc[:, np.r_[0,2:6,8,12:17]]

df.head()
def feature_rel_plot(data, kind, var1=None, var2=None, var3=None, var4=None, var5=None, col=0):

    '''

    Creates a plot that shows relationship between features in a dataframe



    Inputs:



    data:-          dataframe

    kind:-          str; the kind of plot being made ("count",“point”, “bar”, “strip”, “swarm”,

                    “box”, “violin”, or “boxen”, etc)

    var1,var2,...   :-  the columns/variables to be plotted



    '''

    f = plt.figure(figsize=(15,15));

    sns.set_style("darkgrid");



    if kind == 'count':

        [x,hue,col,row] = [var1,var2,var3,var4]

        ax = sns.catplot(data=data, kind=kind, x=x, y=None, hue=hue, col=col, row=row, height=6, aspect=1.3);

        ax.set(ylabel='Number of appointments');



    else:

        [x,y,hue,col,row] = [var1,var2,var3,var4,var5]

        ax = sns.catplot(data=data, kind=kind, x=x, y=y, hue=hue, col=col, row=row, height=6, aspect=1.3);



    features = list(filter(lambda x: x is not None, [var1,var2,var3,var4,var5]))

    ax.fig.subplots_adjust(top=0.9);

    ax.fig.suptitle((f"Relationship Between Number of Appointments and ({', '.join(features)})").upper(), fontsize=16);



    return
df.groupby('gender').gender.count()
# it would be nice to see the above numbers visually

feature_rel_plot(df, 'count', 'gender')
gender_percent = df.groupby('gender').gender.count()*100/df.gender.count()

print(f'Women amount to {gender_percent.Female:.0f}% while men amount to {gender_percent.Male:.0f}%')
pat_by_gender = df.groupby(['patient_id', 'gender']).count().age
pat_by_gender = pat_by_gender.reset_index()
pat_by_gender.dropna(inplace=True)

pat_by_gender = pat_by_gender.groupby('gender').age.count()

pat_by_gender
pat_by_gender.sum() == df.patient_id.nunique()
percent = pat_by_gender/pat_by_gender.sum()

print(f' The dataset contains {percent.Male:.0%} male patients and {percent.Female:.0%} female patients.')
# To visualise this with a bar chart

pat_by_gender.plot(kind='bar')

plt.ylabel('Number of patients')

plt.title('Number of patients by gender')
# visualise with pie chart

pat_by_gender.plot(kind='pie', figsize=(6,7))

plt.title('Number of patients by gender')

plt.ylabel(None);
gs_counts = df.groupby(['status', 'gender']).status.count()

gs_counts
# Let's visualise this

feature_rel_plot(df, 'count', 'status', 'gender')
total_gender = df.groupby('gender').count().iloc[:, 0]

proportions = gs_counts/total_gender

proportions = proportions.reset_index().rename({0:'proportions'}, axis=1)

proportions
# Let's visualise the proportions

feature_rel_plot(proportions.reset_index(), 'bar', 'status', 'proportions', 'gender')
df.waiting_days.describe()
df.groupby('wait_category').count().iloc[:,0]
feature_rel_plot(df, 'count', 'wait_category')
# Show and No Show counts for each wait_category grouping

df.groupby(['wait_category']).status.value_counts()
# Plot

feature_rel_plot(df, 'count', 'wait_category', 'status')
# I'll switch up the axes to get a better view of the trend

df.groupby(['status']).wait_category.value_counts()
# Plot the above

feature_rel_plot(df, 'count', 'status', 'wait_category')
df.groupby(['wait_category','status']).gender.value_counts()
feature_rel_plot(df, 'count', 'wait_category', 'status', 'gender')
# I'll switch up the axes again as before to get a better view

feature_rel_plot(df, 'count', 'status', 'wait_category', 'gender')
df.groupby(['wait_category']).age.mean()
feature_rel_plot(df, 'bar', 'wait_category', 'age')
# histogram showing the age distribution with respect to status(show or no show).

df.groupby('status').age.hist(label=['Show', 'No Show'])

plt.title('Distribution of Patients by Age ')

plt.ylabel('Number of appointments')

plt.xlabel('Age')

plt.legend()
# Overall age distribution with a countplot comparing the Show and No Show trends

feature_rel_plot(df, 'count', 'age', 'status')
feature_rel_plot(df, 'box', 'age')
feature_rel_plot(df, 'count', 'age_range')
# Here are the actual values from the value counts in the bar chart above

df.age_range.value_counts()
df.groupby('gender').age.mean()
feature_rel_plot(df, 'bar', 'gender', 'age')
df.groupby('status').age_range.value_counts()
feature_rel_plot(df, 'count', 'age_range', 'status')
df.sms_received.value_counts()
# visualise the numbers above

feature_rel_plot(df, 'count', 'sms_received')
# Here are the actual numbers

df.groupby('status').sms_received.value_counts()
# visualisation of the above numbers

feature_rel_plot(df, 'count', 'status', 'sms_received')
# number of hypertensive and non-hypertensive patients for both show and no show categories

df.groupby('status').hypertension.value_counts()
# plot to visualise the numbers above

feature_rel_plot(df, 'count', 'hypertension')
df.groupby('hypertension').age.mean()
# visualise the mean ages

feature_rel_plot(df, 'bar', 'hypertension', 'age')
feature_rel_plot(df, 'count', 'hypertension', 'status')