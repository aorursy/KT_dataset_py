# Import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#define function to determine the percentage of patients not attending appointments by certain feature

def per_noshow(DataFrame,feature,feature_value):

    """

    Take in the feature to analyse and the value of that feature and returns the percentage 

    of no-shows. Intended to use as part of lambda function to update DataFrame.

    

    Parameters

    ----------

    df : the DataFrame to analyse

    feature : str

        feature of the DataFrame to analyse

    feature_value : scalar

        the feature_value to analyse

        

    """

    no_show = DataFrame[(DataFrame['No-show']=='Yes') & (DataFrame[feature]==feature_value)]['No-show'].count()

    total = DataFrame[DataFrame[feature]==feature_value][feature].count()

    return round((no_show / total) * 100,2)
# Define a class in for creating graphs.



class feat_to_analyse(object):

    """A feature from the dataframe can have a countplot and percentage of patients not

    attending appointments calculated.

    

    Attributes:

        feature: feature of the dataframe to analyse

    

    """

    def __init__(self,feature):

        self.feature = feature

    

    def graph(self,df):

        """Plot a countplot using the feature assigned from the DataFrame  

        

        Attributes:

            df: the DataFrame where the feature is

        

        """

        sns.countplot(self.feature,data=df,hue='No-show',palette='viridis')



    def percentage(self,df,x):

        """Calculate percentage of patients not attending appointments

        

        Attributes:

            df: the DataFrame where the feature is

            x: return values from the DataFrame based on a certain value

        

        """

        percentage = (sum((df[self.feature]==x) & (df['No-show']=='Yes'))/sum(df[self.feature]==x))*100

        print('Percentage of {} patients not attending appointments: {}%'.format(self.feature,round(percentage,2)))
# import data into dataframe

df = pd.read_csv('../input/KaggleV2-May-2016.csv')
#Check info about dataframe

df.info()
#Rename columns with spelling errors

df.rename(columns={'Hipertension':'Hypertension','Handcap':'Handicap'},inplace=True)
# Grab info about DataFrame

df.describe()
df.head()
# Check for null values

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Clean ages

df = df[(df['Age']>=0) & (df['Age']<100)]
# Drop columns

df.drop(labels=['PatientId','AppointmentID','AppointmentDay'],axis=1,inplace=True)
# confirm

df.head()
# No shows for the entire population

df['No-show'].value_counts()
#Percentage no-shows over entire population

percentage = (df[df['No-show']=='Yes']['No-show'].count()/df['No-show'].count())*100

print('Percentage of patients not attending appointments: {}%'.format(round(percentage,2)))
feature = feat_to_analyse('Gender')

feature.graph(df)

feature.percentage(df,'M')

feature.percentage(df,'F')
feature = feat_to_analyse('Diabetes')

feature.graph(df)

feature.percentage(df,1)
feature = feat_to_analyse('Hypertension')

feature.graph(df)

feature.percentage(df,1)
feature = feat_to_analyse('Alcoholism')

feature.graph(df)

feature.percentage(df,1)
feature = feat_to_analyse('Handicap')

feature.graph(df)

feature.percentage(df,1)

feature.percentage(df,2)

feature.percentage(df,3)

feature.percentage(df,4)
for i in range(0,5):

    print(df[df['Handicap']==i]['Handicap'].value_counts())
feature = feat_to_analyse('Scholarship')

feature.graph(df)

feature.percentage(df,1)
#Change scheduled day to datetime

df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['Time'] = df['ScheduledDay'].apply(lambda time: time.time)

df['Hour'] = df['ScheduledDay'].apply(lambda time: time.hour)

df['Month'] = df['ScheduledDay'].apply(lambda time: time.month)

df['Day of Week'] = df['ScheduledDay'].apply(lambda time: time.dayofweek)
# Map days of the week to numbers

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

df['Day of Week'] = df['Day of Week'].map(dmap)
df.head()
df['Day of Week'].value_counts()
sns.countplot(x='Day of Week',data=df,hue='No-show',

              order=['Mon','Tue','Wed','Thu','Fri','Sat'],palette='viridis')
sns.countplot(x='Hour',data=df,hue='No-show',palette='viridis')
df_time = pd.DataFrame(df['Hour'].value_counts())
df_time.sort_index(inplace=True)

df_time.reset_index(inplace=True)

df_time.rename(columns={'Hour':'Count','index':'Hour'},inplace=True)
df_time['No-Show Percent'] = df_time['Hour'].apply(lambda x: per_noshow(df,'Hour',x))
df_time
plt.figure(figsize=(10,6))

plt.xticks(rotation=45)

sns.barplot(x='Hour',y='No-Show Percent',data=df_time,color='red')
def set_age_group(x):

    """Set an age range

    """

    if x <=3:

        return 'infant',0

    elif x <=10:

        return 'child',1

    elif x <=18:

        return 'adolescent',2

    elif x <= 25:

        return 'Young Adult',3

    elif x <= 45:

        return 'Adult',4

    elif x <= 75:

        return 'Middle Aged',5

    else:

        return 'Elderly',6
df['Age_Range'] = df['Age'].apply(lambda x: set_age_group(x))
df.head()
df['Age_Range'].value_counts()
# Create a DataFrame with value counts of patients ages

df_ages = pd.DataFrame(df['Age_Range'].value_counts())
# Clean up DataFrame

df_ages.reset_index(inplace=True)

df_ages['Description'],df_ages['Rank'] = zip(*df_ages['index'])

df_ages.set_index('Rank',inplace=True)

df_ages.sort_index(inplace=True)

df_ages.rename(columns={'Age_Range':'Count'},inplace=True)
df_ages
# Add a column of the percentage of patients No-show

df_ages['No-Show Percent'] = df_ages['index'].apply(lambda x: per_noshow(df,'Age_Range',x))
df_ages.head()
df_ages.describe()
plt.figure(figsize=(12,6))

plt.xticks(rotation=45)

sns.barplot(x='Description',y='No-Show Percent',data=df_ages,color='red')
# Create a DataFrame with just male patients

df_ages_M_initial = df[df['Gender']=='M']

#Apply same processing as previosuly

df_ages_M = pd.DataFrame(df_ages_M_initial['Age'].value_counts())

df_ages_M.sort_index(inplace=True)

df_ages_M.reset_index(inplace=True)

df_ages_M.rename(columns={'Age':'Male','index':'Age'},inplace=True)

df_ages_M['No-Show Percent Male'] = df_ages_M['Age'].apply(lambda x: per_noshow(df_ages_M_initial,'Age',x))

df_ages_M.head()
# Do the same for female patients

df_ages_F_initial = df[df['Gender']=='F']

df_ages_F = pd.DataFrame(df_ages_F_initial['Age'].value_counts())

df_ages_F.sort_index(inplace=True)

df_ages_F.reset_index(inplace=True)

df_ages_F.rename(columns={'Age':'Female','index':'Age'},inplace=True)

df_ages_F['No-Show Percent Female'] = df_ages_F['Age'].apply(lambda x: per_noshow(df_ages_F_initial,'Age',x))
df_ages_F.head()

df_ages_Combined = df_ages_F.merge(df_ages_M,on='Age')
df_ages_Combined.head()
plt.figure(figsize=(20,5))

ax = df_ages_Combined.plot(x='Age',y='No-Show Percent Female',figsize=(18,6),kind='bar',color='red')

df_ages_Combined.plot(x='Age',y='No-Show Percent Male',ax=ax,kind='bar',color='blue')
# Add a new column to assess whether 

df_ages_Combined['Higher or Lower'] = df_ages_Combined['Age'].apply(lambda x : 

    df_ages_Combined['No-Show Percent Male'].loc[x]-df_ages_Combined['No-Show Percent Female'].iloc[x])
# Define a function to return whether number is positive or negative.

def M_or_F(x):

    """if x is positve return male, if negative return female

    """

    if x > 0:

        return 'M'

    elif x == 0:

        return 'N'

    else:

        return 'F'
df_ages_Combined['Higher or Lower Bin'] = df_ages_Combined['Higher or Lower'].apply(lambda x : M_or_F(x))
df_ages_Combined['Higher or Lower Bin'].value_counts()
df_ages_Combined['Higher or Lower Bin'].loc[18:25].value_counts()
df_ages_Combined['Higher or Lower Bin'].loc[45:75].value_counts()