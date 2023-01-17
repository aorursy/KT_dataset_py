# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_victim_of_rape = pd.read_csv("../input/20_Victims_of_rape.csv")
# Looking at the shape of data

df_victim_of_rape.shape
# Let's understand the dtypes for this data set!

df_victim_of_rape.dtypes
df_victim_of_rape.head()
# Let's see what are the AREAs, provided to us for exploration!

df_victim_of_rape.Area_Name.value_counts()
df_victim_of_rape.groupby(['Year','Subgroup']).Rape_Cases_Reported.sum()
plt.figure(figsize=(8,8))

df_victim_of_rape.groupby(['Year','Subgroup']).Rape_Cases_Reported.sum().plot(kind='bar');

#We will look at each subgroup of rape over the years and will infer!
df_victim_of_rape.Rape_Cases_Reported.plot(kind='hist',bins=20);
up_total_rape = df_victim_of_rape.loc[df_victim_of_rape['Area_Name']=='Uttar Pradesh']

up_victim_2010_total = up_total_rape [(up_total_rape['Year']==2010) & (up_total_rape['Subgroup']=='Total Rape Victims')]

up_victim_2010_total_incest_rape = up_total_rape [(up_total_rape['Year']==2010) & (up_total_rape['Subgroup']=='Victims of Incest Rape')]



#Plotting age breakup of victims

ax = up_victim_2010_total[['Victims_Upto_10_Yrs','Victims_Between_10-14_Yrs','Victims_Between_14-18_Yrs','Victims_Between_18-30_Yrs','Victims_Between_30-50_Yrs']].plot(kind='bar',legend=True, title = 'Age Breakup of rape victims (Uttar Pradesh..!!)')

ax.set_ylabel("No of Victims", fontsize=12)

ax.set_xticklabels([]);

ax = up_victim_2010_total_incest_rape[['Victims_Upto_10_Yrs','Victims_Between_10-14_Yrs','Victims_Between_14-18_Yrs','Victims_Between_18-30_Yrs','Victims_Between_30-50_Yrs']].plot(kind='bar',legend=True, title = 'Age Breakup of Incest rape victims (Uttar Pradesh..!!)')

ax.set_ylabel("No of Victims", fontsize=12)

ax.set_xticklabels([]);
victims_rape_2010_total = df_victim_of_rape[(df_victim_of_rape['Year']==2010) & (df_victim_of_rape['Subgroup']== 'Total Rape Victims')]

ax1 = victims_rape_2010_total['Victims_of_Rape_Total'].plot(kind='barh',figsize=(20, 15))

ax1.set_xlabel("Number of rape victims (2010)", fontsize=25)

ax1.set_yticklabels(victims_rape_2010_total['Area_Name']);
import seaborn as sns

plt.figure(figsize=(8,8))

df_corr=df_victim_of_rape.corr()

sns.heatmap(df_corr, xticklabels = df_corr.columns.values, yticklabels = df_corr.columns.values,annot=True);
# What is the mean of the Total Rapes:

df_mean = df_victim_of_rape.Victims_of_Rape_Total.mean()

print(df_mean)
# We have seen the MEAN value, now we will try to see the states where the Total Rapes are greater than MEAN

df_total = df_victim_of_rape[df_victim_of_rape.Victims_of_Rape_Total>362]

df_total.head()
# Let's find out those States! where the Rapes are more than Mean!

plt.figure(figsize=(24,5))

plt.title('Count of States, where the Rapes are more than Mean Value of Rape!');

sns.countplot(df_total.Area_Name);

plt.xticks(rotation = 60);
# Let's explore the recieved Data Frame and perform drilling analysis on that!

data_frame_to_drill = df_total.copy()
data_frame_to_drill.describe()
data_frame_to_drill.shape

# We have only 326 Rows to Analyze, so let's break it down!
data_frame_to_drill.Victims_Upto_10_Yrs.sum()

# This is the sum of Victims which are upto 10 Years in age only!, but over the due course of time!
plt.figure(figsize=(24,8))

sns.barplot(x = data_frame_to_drill.Area_Name, y = data_frame_to_drill['Rape_Cases_Reported'], color = "red")

upto_18 = sns.barplot(x = data_frame_to_drill.Area_Name, y = data_frame_to_drill['Victims_Between_14-18_Yrs'], color = "#a8ddb5")

topbar = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none')

upto_18 = plt.Rectangle((0,0),1,1,fc='#a8ddb5',  edgecolor = 'none')

l = plt.legend([upto_18, topbar], ['Victims between 14 to 18 years of Age', 'Total Rape Cases'], loc=1, ncol = 2, prop={'size':16})

l.draw_frame(False)

sns.despine(left=True)
plt.figure(figsize=(24,8))

sns.barplot(x = data_frame_to_drill.Area_Name, y = data_frame_to_drill['Rape_Cases_Reported'], color = "red")

upto_30 = sns.barplot(x = data_frame_to_drill.Area_Name, y = data_frame_to_drill['Victims_Between_18-30_Yrs'], color = "#fe9929")

topbar = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none')

upto_30 = plt.Rectangle((0,0),1,1,fc='#fe9929',  edgecolor = 'none')

l = plt.legend([upto_30, topbar], ['Victims between 18 to 30', 'Total Rape Cases'], loc=1, ncol = 2, prop={'size':16})

l.draw_frame(False)

sns.despine(left=True)

#upto_30.set_ylabel("Y-axis label");

#upto_30.set_xlabel("X-axis label");
plt.figure(figsize=(24,8))

sns.boxplot(data_frame_to_drill.Victims_Upto_10_Yrs,data_frame_to_drill.Area_Name,data = data_frame_to_drill);

plt.title('Spread of Rapes performed on Children upto 10 Years!');
df_31s_f = pd.read_csv("../input/31_Serious_fraud.csv")
df_31s_f.describe()
df_31s_f.dtypes
df_31s_f.shape

#We have 448 rows and 9 Columns
df_31s_f.head(10)
df_31s_f.isnull().sum()
df_31s_f.Loss_of_Property_1_10_Crores.mean()
df_31s_f.groupby(['Area_Name','Year']).Loss_of_Property_10_25_Crores.plot(kind='barh');

plt.xlabel('Bins')

plt.ylabel('Loss of Property')

plt.show()
#1. Gujarat

df_plot_fraud = df_31s_f.loc[df_31s_f['Area_Name'] == 'Gujarat']

df_year_fraud = df_plot_fraud[(df_plot_fraud['Year'] == 2001) & (df_plot_fraud['Group_Name'] == 'Serious Fraud - Cheating')]

ax = df_year_fraud[['Loss_of_Property_1_10_Crores','Loss_of_Property_10_25_Crores','Loss_of_Property_25_50_Crores','Loss_of_Property_50_100_Crores','Loss_of_Property_Above_100_Crores']].plot(kind='bar',legend=True, title = 'Breakup of Frauds Performed in Gujrat')

ax.set_ylabel("Loss of Property", fontsize=12)

ax.set_xticklabels([]);
#2. Delhi

df_plot_fraud = df_31s_f.loc[df_31s_f['Area_Name'] == 'Delhi']

df_year_fraud = df_plot_fraud[(df_plot_fraud['Year'] == 2001) & (df_plot_fraud['Group_Name'] == 'Serious Fraud - Cheating')]

ax = df_year_fraud[['Loss_of_Property_1_10_Crores','Loss_of_Property_10_25_Crores','Loss_of_Property_25_50_Crores','Loss_of_Property_50_100_Crores','Loss_of_Property_Above_100_Crores']].plot(kind='bar',legend=True, title = 'Breakup of Frauds Performed in Gujrat')

ax.set_ylabel("Loss of Property", fontsize=15)

ax.set_xticklabels([]);
#3. Andhra Pradesh

df_plot_fraud = df_31s_f.loc[df_31s_f['Area_Name'] == 'Andhra Pradesh']

df_year_fraud = df_plot_fraud[(df_plot_fraud['Year'] == 2001) & (df_plot_fraud['Group_Name'] == 'Serious Fraud - Cheating')]

ax = df_year_fraud[['Loss_of_Property_1_10_Crores','Loss_of_Property_10_25_Crores','Loss_of_Property_25_50_Crores','Loss_of_Property_50_100_Crores','Loss_of_Property_Above_100_Crores']].plot(kind='bar',legend=True, title = 'Breakup of Frauds Performed in Gujrat')

ax.set_ylabel("Loss of Property", fontsize=12)

ax.set_xticklabels([]);
df_frauds_1_10_crores  = np.where(df_31s_f['Loss_of_Property_1_10_Crores']>40.0)

print(df_frauds_1_10_crores)

for i in df_frauds_1_10_crores:

    print(df_31s_f['Area_Name'][i])
sns.kdeplot(df_31s_f['Loss_of_Property_1_10_Crores'],shade = True,color="red",alpha = 0.4)

sns.kdeplot(df_31s_f['Loss_of_Property_10_25_Crores'],shade = True,color="blue",alpha = 0.3)

sns.kdeplot(df_31s_f['Loss_of_Property_25_50_Crores'],shade = True,color="orange",alpha = 0.2)

plt.show();
p = sns.countplot(x="Loss_of_Property_1_10_Crores" , data=df_31s_f , palette = "bright")

_ = plt.setp(p.get_xticklabels(),rotation = 90)
p = sns.countplot(x="Loss_of_Property_10_25_Crores" , data=df_31s_f , palette = "bright")

_ = plt.setp(p.get_xticklabels(),rotation = 90)
trials_by_court = pd.read_csv('../input/29_Period_of_trials_by_courts.csv')

trials_by_court.head(10)
print('Shape of our Data frame: ',trials_by_court.shape)

print("-----------------")

print(trials_by_court.info())
# Let's find the Missing values:

print(round(100*(trials_by_court.isnull().sum()/len(trials_by_court)),2))
# Let's plot a HeatMap for visualization of Missing values;

sns.heatmap(trials_by_court.isnull(),cbar=False);
# So we see that maximum of 8% is the missing value faced, hence we can drop the rows with missing values, and this will help us in obtaining more precise data!

trials_by_court = trials_by_court.dropna(axis=0)

trials_by_court.info()
# So approximately 200 rows are dropped! and We feel confident enough to move onto!

sns.heatmap(trials_by_court.isnull(),cbar=False);
trials_by_court.Area_Name.value_counts()
trials_by_court.Group_Name.value_counts()
trials_by_court.Sub_Group_Name.value_counts()
trials_by_court.Year.value_counts()

# We will group the Years into batch of 3 Years: 2004 to 2007 and 2008 to 2010
# We also see that we have same Sub Group Name and Group Name, reflecting the same variable, hence we will drop one of this variable.

# Let's bin first:

mapping = {2004 : '2004 to 2007',2005 : '2004 to 2007',2006 : '2004 to 2007',2007 : '2004 to 2007',

           2008 : '2008 to 2010',2009 : '2008 to 2010',2010 : '2008 to 2010'}

trials_by_court['Year'] = trials_by_court['Year'].apply(mapping.get)

trials_by_court.head(50)
trials_by_court.Year.value_counts()
# Let's look at the distribution of the Year categories on the period of trials as TOTAL!

sns.barplot(x = 'Year' , y = 'PT_Total',data = trials_by_court);
# Pairplot

plt.figure(figsize=(20,20));

sns.pairplot(trials_by_court);
plt.figure(figsize=(12,5));

# Let's see the affect of the Trial period, less than 6 months. This will give the duration where the cases have been solved quickly

sns.barplot(x = trials_by_court.PT_Less_than_6_Months,y = trials_by_court.Year,data=trials_by_court);
# next is we can see the Areas where Trial period of less than 6 months has happened. we will do so by taking the mean of that variables and then returning out the true dataframe

mean_pt_less_than_6_months = trials_by_court.PT_Less_than_6_Months.mean()

mean_pt_less_than_6_months
df_less_than_6_months = trials_by_court.loc[trials_by_court.PT_Less_than_6_Months >= mean_pt_less_than_6_months]

df_less_than_6_months.head()
print(df_less_than_6_months.shape)

print(df_less_than_6_months.Area_Name.value_counts())
# Let's try plotting for these areas:

plt.figure(figsize=(8,8));

plt.title('Plot of Area vs Number of Trials!');

plt.rcParams["axes.labelsize"] = 20

plt.yticks(rotation=15)

sns.barplot(x = trials_by_court.index,y=df_less_than_6_months.Area_Name,data = df_less_than_6_months,ci=None);

plt.xlabel('Frequency!');

# Here we have used the count of our original dataset, as we want to see the frequency of the count of the states where Trials have been executed within 6 months.

# This was tricky to think, as i spent around 25 minutes to see this in a way!
trials_by_court = trials_by_court.drop('Sub_Group_Name',axis=1)

trials_by_court.info()
plt.figure(figsize=(8,8))

sns.barplot(x = 'Year' , y='PT_Less_than_6_Months' , hue = 'Group_Name' , data = trials_by_court,ci=None);