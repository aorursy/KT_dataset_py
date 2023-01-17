# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline
# Load data
df = pd.read_csv('../input/KaggleV2-May-2016.csv')
df.head()
# Check size of dataframe
df.shape
# Info about the data
df.info()
# General descriptive summary about the data
df.describe()
# Create a copy of data to work with
# Copied data can be manipulated and changed as per the requirement of analysis
# Original data source remains unchanged and available at all times.
df_med_app = df.copy()
df_med_app.head()
# Checking for null values if any
sum(df_med_app.isnull().any())
# Checking for duplicate rows in data
sum(df_med_app.duplicated())
# Checking data types of all columns
df_med_app.dtypes
# Get descriptive summary of data
df_med_app.describe()
# Drop the columns PatientId and AppointmentID
df_med_app.drop(['PatientId', 'AppointmentID'], axis=1, inplace=True)
# Check whether the columns have been dropped successfully
df_med_app.head(2)
# Correcting spelling names
df_med_app.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap'}, inplace=True)
# confirm changes
df_med_app.head(2)
df_med_app['ScheduledDay'] = pd.to_datetime(df_med_app.ScheduledDay)
df_med_app['AppointmentDay'] = pd.to_datetime(df_med_app.AppointmentDay)
# check the changed data types
df_med_app[['ScheduledDay', 'AppointmentDay']].dtypes
df_med_app[['ScheduledDay', 'AppointmentDay']].head(2)
df_med_app['ScheduledDay'] = df_med_app['ScheduledDay'].dt.normalize()
# Check both columns - ScheduledDay and AppointmentDay after normalization
df_med_app[['ScheduledDay', 'AppointmentDay']].head(2)
df_med_app[df_med_app['AppointmentDay'] < df_med_app['ScheduledDay']]
df_med_app['Days_To_Wait'] = df_med_app['AppointmentDay'] - df_med_app['ScheduledDay']
df_med_app['Days_To_Wait'].head(2)
df_med_app.dtypes
df_med_app['Days_To_Wait'] = (df_med_app['Days_To_Wait'].apply(lambda x: str(x))).apply(lambda x: x.split(" ")[0])
df_med_app['Days_To_Wait'] = df_med_app['Days_To_Wait'].apply(lambda x: int(x))
# Check the changes for 'Days_To_Wait' column values
df_med_app.dtypes
df_med_app['Days_To_Wait'].min(), df_med_app['Days_To_Wait'].max()
# Drop rows for which 'Days_To_Wait' value is negative
df_med_app.drop(df_med_app[df_med_app['Days_To_Wait'] < 0].index, inplace=True)
# Verify there are no more columns for which 'Days_To_Wait' value is negative
df_med_app[df_med_app['Days_To_Wait'] < 0]
# Check the unique values for 'Scholarship'
df_med_app.Scholarship.unique()
# Check the unique values for 'Hypertension'
df_med_app.Hypertension.unique()
# Check the unique values for 'Diabetes'
df_med_app.Diabetes.unique()
# Check the unique values for 'Alcoholism'
df_med_app.Alcoholism.unique()
# Check the unique values for 'SMS_received'
df_med_app.SMS_received.unique()
# Check the unique values for 'Handicap'
df_med_app.Handicap.unique()
# Using the loc function which takes index (row label as index) as an argument
df_med_app.loc[df_med_app[df_med_app.Handicap > 0].index, 'Handicap'] = 1
# Checking the unique values for handicap column
df_med_app.Handicap.unique()
# Rows containing negative age values
df_med_app[df_med_app.Age < 0]
# Using loc function which takes label (index label) as an argument to update with mean value
df_med_app.loc[99832, 'Age'] = df_med_app.Age.mean()
# Cross check for any negative values
df_med_app[df_med_app.Age < 0]
# Using the loc function which takes index (row label as index) as an argument
df_med_app.loc[df_med_app[df_med_app['No-show'] == 'Yes'].index, 'No-show'] = 1
df_med_app.loc[df_med_app[df_med_app['No-show'] == 'No'].index, 'No-show'] = 0
df_med_app.dtypes
# Change string values for column 'No-show' to integer values
df_med_app['No-show'] = df_med_app['No-show'].apply(lambda x: int(x))
# Check data types
df_med_app.dtypes
# Verify the change has been made correctly
df_med_app[df_med_app['No-show'] == 1].shape[0] + \
df_med_app[df_med_app['No-show'] == 0].shape[0] == df_med_app.shape[0]
# Changing column names to lower case characters
# replace hyphens with underscores
df_med_app.rename(columns=lambda x: x.strip().lower().replace("-", "_"), inplace=True)
# confirm changes
df_med_app.head()
no_show = df_med_app.no_show == 1
show_up = df_med_app.no_show == 0
# Get the count of no_show and appointment_completed after classifying the data into these 2 categories
df_med_app[no_show].shape[0], df_med_app[show_up].shape[0]
# Get the proportion of no_show and appointment_completed
df_med_app[no_show].shape[0] / df_med_app.shape[0], df_med_app[show_up].shape[0] / df_med_app.shape[0]
# Checking for accuracy of proportions
df_med_app[no_show].shape[0] + df_med_app[show_up].shape[0] == df_med_app.shape[0]
# Pie graph for proportion of appointments completed and no-shows
sns.set(style='darkgrid')
plt.pie(df_med_app.no_show.value_counts(), \
        labels=['Show_up', 'No_shows'], \
        explode=[0.1, 0], autopct="%.2f%%")
plt.axis('equal');
# Pie graph for proportion of male and female
plt.pie(df_med_app.gender.value_counts(), \
        labels=['Female', 'Male'], \
        autopct="%.2f%%")
plt.axis('equal');
# Find the poportion of females in the categories 'no_show' and 'appointment_completed' using masks
bar_labels = ['No show', 'Show up']
x_pos = list(range(len(bar_labels)))
proportion_female_no_show = df_med_app[no_show].query('gender == "F"').shape[0] / df_med_app[no_show].shape[0]
proportion_female_show_up = df_med_app[show_up].query('gender == "F"').shape[0] / df_med_app[show_up].shape[0]
proportion_vals_female = proportion_female_no_show, proportion_female_show_up
proportion_vals_female
# plotting the bar chart
plt.bar(x_pos, proportion_vals_female, align='center', alpha=0.5)
plt.grid()
plt.ylabel('Proportions')
plt.xticks(x_pos, bar_labels)
plt.title('Proportion of females for categories - no show and show up');
# Find the poportion of males in the categories 'no_show' and 'appointment_completed' using masks
bar_labels = ['No show', 'Show up']
x_pos = list(range(len(bar_labels)))
proportion_male_no_show = df_med_app[no_show].query('gender == "M"').shape[0] / df_med_app[no_show].shape[0]
proportion_male_show_up = df_med_app[show_up].query('gender == "M"').shape[0] / df_med_app[show_up].shape[0]
proportion_vals_male = proportion_male_no_show, proportion_male_show_up
proportion_vals_male
# plotting the bar chart
plt.bar(x_pos, proportion_vals_male, align='center', alpha=0.5)
plt.grid()
plt.ylabel('Proportions')
plt.xticks(x_pos, bar_labels)
plt.title('Proportion of males for categories - no show and show up');
scholarship = df_med_app.scholarship == 1
no_scholarship = df_med_app.scholarship == 0
# Get the count of scholarship and no_scholarship after classifying the data into these 2 categories
df_med_app[scholarship].shape[0], df_med_app[no_scholarship].shape[0]
# Get the proportion of scholarship and no_scholarship
df_med_app[scholarship].shape[0] / df_med_app.shape[0], df_med_app[no_scholarship].shape[0] / df_med_app.shape[0]
# Checking for accuracy of proportions
df_med_app[scholarship].shape[0] + df_med_app[no_scholarship].shape[0] == df_med_app.shape[0]
bar_labels = ['No show', 'Show up']
x_pos = list(range(len(bar_labels)))
proportion_scholarship_no_show = df_med_app[no_show].query('scholarship == 1').shape[0] / df_med_app[no_show].shape[0]
proportion_scholarship_show_up = df_med_app[show_up].query('scholarship == 1').shape[0] / df_med_app[show_up].shape[0]
proportion_vals_scholarship = proportion_scholarship_no_show, proportion_scholarship_show_up
proportion_vals_scholarship
plt.bar(x_pos, proportion_vals_scholarship, align='center', alpha=0.5)
plt.grid()
plt.ylabel('Proportions')
plt.xticks(x_pos, bar_labels)
plt.title('Scholarship proportions for categories - no show and show up');
bar_labels = ['No show', 'Show up']
x_pos = list(range(len(bar_labels)))
proportion_no_scholarship_no_show = df_med_app[no_show].query('scholarship == 0').shape[0] / df_med_app[no_show].shape[0]
proportion_no_scholarship_show_up = df_med_app[show_up].query('scholarship == 0').shape[0] / df_med_app[show_up].shape[0]
proportion_vals_no_scholarship = proportion_no_scholarship_no_show, proportion_no_scholarship_show_up
proportion_vals_no_scholarship
plt.bar(x_pos, proportion_vals_no_scholarship, align='center', alpha=0.5)
plt.grid()
plt.ylabel('Proportions')
plt.xticks(x_pos, bar_labels)
plt.title('No scholarship proportions for categories - no show and show up');
hypertension = df_med_app.hypertension == 1
no_hypertension = df_med_app.hypertension == 0
# Get the count of hypertension and no_hypertension after classifying the data into these 2 categories
df_med_app[hypertension].shape[0], df_med_app[no_hypertension].shape[0]
# Get the proportion of hypertension and no_hypertension
df_med_app[hypertension].shape[0] / df_med_app.shape[0], df_med_app[no_hypertension].shape[0] / df_med_app.shape[0]
# Checking for accuracy of proportions
df_med_app[hypertension].shape[0] + df_med_app[no_hypertension].shape[0] == df_med_app.shape[0]
bar_labels = ['No show', 'Show up']
x_pos = list(range(len(bar_labels)))
proportion_hypertension_no_show = df_med_app[no_show].query('hypertension == 1').shape[0] / df_med_app[no_show].shape[0]
proportion_hypertension_show_up = df_med_app[show_up].query('hypertension == 1').shape[0] / df_med_app[show_up].shape[0]
proportion_vals_hypertension = proportion_hypertension_no_show, proportion_hypertension_show_up
proportion_vals_hypertension
plt.bar(x_pos, proportion_vals_hypertension, align='center', alpha=0.5)
plt.grid()
plt.ylabel('Proportions')
plt.xticks(x_pos, bar_labels)
plt.title('Hypertension proportions for categories - no show and show up');
bar_labels = ['No show', 'Show up']
x_pos = list(range(len(bar_labels)))
proportion_no_hypertension_no_show = df_med_app[no_show].query('hypertension == 0').shape[0] / df_med_app[no_show].shape[0]
proportion_no_hypertension_show_up = df_med_app[show_up].query('hypertension == 0').shape[0] / df_med_app[show_up].shape[0]
proportion_vals_no_hypertension = proportion_no_hypertension_no_show, proportion_no_hypertension_show_up
proportion_vals_no_hypertension
plt.bar(x_pos, proportion_vals_no_hypertension, align='center', alpha=0.5)
plt.grid()
plt.ylabel('Proportions')
plt.xticks(x_pos, bar_labels)
plt.title('No hypertension proportions for categories - no show and show up');
diabetes = df_med_app.diabetes == 1
no_diabetes = df_med_app.diabetes == 0
# Get the count of diabetes and no_diabetes after classifying the data into these 2 categories
df_med_app[diabetes].shape[0], df_med_app[no_diabetes].shape[0]
# Get the proportion of diabetes and no_diabetes
df_med_app[diabetes].shape[0] / df_med_app.shape[0], df_med_app[no_diabetes].shape[0] / df_med_app.shape[0]
# Checking for accuracy of proportions
df_med_app[diabetes].shape[0] + df_med_app[no_diabetes].shape[0] == df_med_app.shape[0]
bar_labels = ['No show', 'Show up']
x_pos = list(range(len(bar_labels)))
proportion_diabetes_no_show = df_med_app[no_show].query('diabetes == 1').shape[0] / df_med_app[no_show].shape[0]
proportion_diabetes_show_up = df_med_app[show_up].query('diabetes == 1').shape[0] / df_med_app[show_up].shape[0]
proportion_vals_diabetes = proportion_diabetes_no_show, proportion_diabetes_show_up
proportion_vals_diabetes
plt.bar(x_pos, proportion_vals_diabetes, align='center', alpha=0.5)
plt.grid()
plt.ylabel('Proportions')
plt.xticks(x_pos, bar_labels)
plt.title('Diabetes proportions for categories - no show and show up');
bar_labels = ['No show', 'Show up']
x_pos = list(range(len(bar_labels)))
proportion_no_diabetes_no_show = df_med_app[no_show].query('diabetes == 0').shape[0] / df_med_app[no_show].shape[0]
proportion_no_diabetes_show_up = df_med_app[show_up].query('diabetes == 0').shape[0] / df_med_app[show_up].shape[0]
proportion_vals_diabetes = proportion_no_diabetes_no_show, proportion_no_diabetes_show_up
proportion_vals_diabetes
plt.bar(x_pos, proportion_vals_diabetes, align='center', alpha=0.5)
plt.grid()
plt.ylabel('Proportions')
plt.xticks(x_pos, bar_labels)
plt.title('No diabetes proportions for categories - no show and show up');
alcoholism = df_med_app.alcoholism == 1
no_alcoholism = df_med_app.alcoholism == 0
# Get the count of alcoholism and no_alcoholism after classifying the data into these 2 categories
df_med_app[alcoholism].shape[0], df_med_app[no_alcoholism].shape[0]
# Get the proportion of alcoholism and no_alcoholism
df_med_app[alcoholism].shape[0] / df_med_app.shape[0], df_med_app[no_alcoholism].shape[0] / df_med_app.shape[0]
# Checking for accuracy of proportions
df_med_app[alcoholism].shape[0] + df_med_app[no_alcoholism].shape[0] == df_med_app.shape[0]
bar_labels = ['No show', 'Show up']
x_pos = list(range(len(bar_labels)))
proportion_alcoholism_no_show = df_med_app[no_show].query('alcoholism == 1').shape[0] / df_med_app[no_show].shape[0]
proportion_alcoholism_show_up = df_med_app[show_up].query('alcoholism == 1').shape[0] / df_med_app[show_up].shape[0]
proportion_vals_alcoholism = proportion_alcoholism_no_show, proportion_alcoholism_show_up
proportion_vals_alcoholism
plt.bar(x_pos, proportion_vals_alcoholism, align='center', alpha=0.5)
plt.grid()
plt.ylabel('Proportions')
plt.xticks(x_pos, bar_labels)
plt.title('Alcoholism proportions for categories - no show and show up');
bar_labels = ['No show', 'Show up']
x_pos = list(range(len(bar_labels)))
proportion_no_alcoholism_no_show = df_med_app[no_show].query('alcoholism == 0').shape[0] / df_med_app[no_show].shape[0]
proportion_no_alcoholism_show_up = df_med_app[show_up].query('alcoholism == 0').shape[0] / df_med_app[show_up].shape[0]
proportion_vals_no_alcoholism = proportion_no_alcoholism_no_show, proportion_no_alcoholism_show_up
proportion_vals_no_alcoholism
plt.bar(x_pos, proportion_vals_no_alcoholism, align='center', alpha=0.5)
plt.grid()
plt.ylabel('Proportions')
plt.xticks(x_pos, bar_labels)
plt.title('No alcoholism proportions for categories - no show and show up');
handicap = df_med_app.handicap == 1
no_handicap = df_med_app.handicap == 0
# Get the count of handicap and no_handicap after classifying the data into these 2 categories
df_med_app[handicap].shape[0], df_med_app[no_handicap].shape[0]
# Get the proportion of handicap and no_handicap
df_med_app[handicap].shape[0] / df_med_app.shape[0], df_med_app[no_handicap].shape[0] / df_med_app.shape[0]
# Checking for accuracy of proportions
df_med_app[handicap].shape[0] + df_med_app[no_handicap].shape[0] == df_med_app.shape[0]
bar_labels = ['No show', 'Show up']
x_pos = list(range(len(bar_labels)))
proportion_handicap_no_show = df_med_app[no_show].query('handicap == 1').shape[0] / df_med_app[no_show].shape[0]
proportion_handicap_show_up = df_med_app[show_up].query('handicap == 1').shape[0] / df_med_app[show_up].shape[0]
proportion_vals_handicap = proportion_handicap_no_show, proportion_handicap_show_up
proportion_vals_handicap
plt.bar(x_pos, proportion_vals_handicap, align='center', alpha=0.5)
plt.grid()
plt.ylabel('Proportions')
plt.xticks(x_pos, bar_labels)
plt.title('Handicap proportions for categories - no show and show up');
bar_labels = ['No show', 'Show up']
x_pos = list(range(len(bar_labels)))
proportion_no_handicap_no_show = df_med_app[no_show].query('handicap == 0').shape[0] / df_med_app[no_show].shape[0]
proportion_no_handicap_show_up = df_med_app[show_up].query('handicap == 0').shape[0] / df_med_app[show_up].shape[0]
proportion_vals_no_handicap = proportion_no_handicap_no_show, proportion_no_handicap_show_up
proportion_vals_no_handicap
plt.bar(x_pos, proportion_vals_no_handicap, align='center', alpha=0.5)
plt.grid()
plt.ylabel('Proportions')
plt.xticks(x_pos, bar_labels)
plt.title('No handicap proportions for categories - no show and show up');
sms_received = df_med_app.sms_received == 1
sms_not_received = df_med_app.sms_received == 0
# Get the count of sms_received and sms_not_received after classifying the data into these 2 categories
df_med_app[sms_received].shape[0], df_med_app[sms_not_received].shape[0]
# Get the proportion of sms_received and sms_not_received
df_med_app[sms_received].shape[0] / df_med_app.shape[0], df_med_app[sms_not_received].shape[0] / df_med_app.shape[0]
# Checking for accuracy of proportions
df_med_app[sms_received].shape[0] + df_med_app[sms_not_received].shape[0] == df_med_app.shape[0]
bar_labels = ['No show', 'Show up']
x_pos = list(range(len(bar_labels)))
proportion_sms_received_no_show = df_med_app[no_show].query('sms_received == 1').shape[0] / df_med_app[no_show].shape[0]
proportion_sms_received_show_up = df_med_app[show_up].query('sms_received == 1').shape[0] / df_med_app[show_up].shape[0]
proportion_vals_sms_received = proportion_sms_received_no_show, proportion_sms_received_show_up
proportion_vals_sms_received
plt.bar(x_pos, proportion_vals_sms_received, align='center', alpha=0.5)
plt.grid()
plt.ylabel('Proportions')
plt.xticks(x_pos, bar_labels)
plt.title('Sms received proportions for categories - no show and show up');
bar_labels = ['No show', 'Show up']
x_pos = list(range(len(bar_labels)))
proportion_sms_not_received_no_show = df_med_app[no_show].query('sms_received == 0').shape[0] / df_med_app[no_show].shape[0]
proportion_sms_not_received_show_up = df_med_app[show_up].query('sms_received == 0').shape[0] / df_med_app[show_up].shape[0]
proportion_vals_sms_not_received = proportion_sms_not_received_no_show, proportion_sms_not_received_show_up
proportion_vals_sms_not_received
plt.bar(x_pos, proportion_vals_sms_not_received, align='center', alpha=0.5)
plt.grid()
plt.ylabel('Proportions')
plt.xticks(x_pos, bar_labels)
plt.title('Sms not received proportions for categories - no show and show up');
df_med_app['days_to_wait'].describe()
days_to_wait_below_mean = df_med_app['days_to_wait'] < df_med_app['days_to_wait'].mean()
days_to_wait_above_mean = df_med_app['days_to_wait'] > df_med_app['days_to_wait'].mean()
# Get the count of days_to_wait_below_mean and days_to_wait_above_mean after classifying data into these 2 categories
df_med_app[days_to_wait_below_mean].shape[0], df_med_app[days_to_wait_above_mean].shape[0]
# Get the proportion of days_to_wait_below_mean and days_to_wait_above_mean
df_med_app[days_to_wait_below_mean].shape[0] / df_med_app.shape[0], df_med_app[days_to_wait_above_mean].shape[0] / df_med_app.shape[0]
# Checking for accuracy of proportions
df_med_app[days_to_wait_below_mean].shape[0] + df_med_app[days_to_wait_above_mean].shape[0] == df_med_app.shape[0]
bar_labels = ['No show', 'Show up']
x_pos = list(range(len(bar_labels)))
proportion_wait_below_mean_no_show = df_med_app[no_show].query('days_to_wait < 10').shape[0] / df_med_app[no_show].shape[0]
proportion_wait_below_mean_show_up = df_med_app[show_up].query('days_to_wait < 10').shape[0] / df_med_app[show_up].shape[0]
proportion_vals_wait_below_mean = proportion_wait_below_mean_no_show, proportion_wait_below_mean_show_up
proportion_vals_wait_below_mean
plt.bar(x_pos, proportion_vals_wait_below_mean, align='center', alpha=0.5)
plt.grid()
plt.ylabel('Proportions')
plt.xticks(x_pos, bar_labels)
plt.title('Days to wait below mean proportions for categories - no show and show up');
bar_labels = ['No show', 'Show up']
x_pos = list(range(len(bar_labels)))
proportion_wait_above_mean_no_show = df_med_app[no_show].query('days_to_wait > 10').shape[0] / df_med_app[no_show].shape[0]
proportion_wait_above_mean_show_up = df_med_app[show_up].query('days_to_wait > 10').shape[0] / df_med_app[show_up].shape[0]
proportion_vals_wait_above_mean = proportion_wait_above_mean_no_show, proportion_wait_above_mean_show_up
proportion_vals_wait_above_mean
plt.bar(x_pos, proportion_vals_wait_above_mean, align='center', alpha=0.5)
plt.grid()
plt.ylabel('Proportions')
plt.xticks(x_pos, bar_labels)
plt.title('Days to wait above mean proportions for categories - no show and show up');
no_show_neighbourhoods = df_med_app.groupby('neighbourhood').sum()['no_show']
top_10_no_show_neighbourhoods = no_show_neighbourhoods.sort_values(ascending=False).head(10)
top_10_no_show_neighbourhoods
# Plot the graph
top_10_no_show_neighbourhoods.plot(kind='bar',figsize=(8, 8),\
                  title = 'Top 10 no show neighbourhoods');
plt.xlabel('Neighbourhoods')
plt.ylabel('No show count')