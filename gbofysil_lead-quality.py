# Import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import *
%matplotlib inline
from sklearn.preprocessing import Imputer

# Settings
pd.set_option('display.max_columns', 770)
# Load data
data = pd.read_csv('../input/leadscore_weekly_conversion_data_20180119.csv')
# Preview data
data.head()
# Only include trials that ended before 2018
completed_trials = (data['trial_end_date'] <= '2018-01-01')
data = data[completed_trials]
#Remove trials with over 20 teachers
non_outlier_teachers = data['total_teacher_count'] < 20
data = data[non_outlier_teachers]
# Create School Features

# Percent of Students By Ethnicity
data['am_percent'] = data['am'] / data['member']
data['as_percent'] = data['as'] / data['member']
data['bl_percent'] = data['bl'] / data['member']
data['wh_percent'] = data['wh'] / data['member']
data['hi_percent'] = data['hi'] / data['member']
data['hp_percent'] = data['hp'] / data['member']
data['tr_percent'] = data['tr'] / data['member']

# Percent of Students on the Free Lunch Program
data['totfrl_percent'] = data['totfrl'] / data['member']
data[['totfrl_percent', 'totfrl', 'member']].describe()

# Student to Teacher Ratio
data['pupfte'] = data['member'] / data['fte']

# Has School Data Dummy Variable
# 30% of data does not have school information

# Create Seasonality Features

# Convert start_date and end_date strings to datetime
data['start_datetime']    = pd.to_datetime(data['start_date'])
data['end_datetime']      = pd.to_datetime(data['end_date'])

# Month
data['start_month']       = data['start_datetime'].dt.month

# Year
data['start_year']        = data['start_datetime'].dt.year

# Day of the Week
data['start_day_of_week'] = data['start_datetime'].dt.dayofweek

# Year-Month for visualizations
data['start_year_month']  = data['start_datetime'].dt.to_period('m')
# Create Site Engagement Features

# Students Per Teacher  (activate_students / active_teachers)
data['students_per_teacher'] = data['active_students']/data['active_teachers']
engagement_data['students_per_teacher'].fillna(0)

# Student Participation Rate (active students / total students)
engagement_data['students_participation'] = engagement_data['active_students']/engagement_data['total_student_count']
engagement_data[engagement_data['total_student_count'] > 1].describe()

# Teacher Participation Rate (active teachers / total teachers)
engagement_data['teacher_participation'] = engagement_data['active_teachers']/engagement_data['total_teacher_count']
# Create Engagement Over Time Features

# Trial Duration in Weeks
# trial duration / 7 to get the number of trial weeks, rounded down to get last full week
data['trial_duration_weeks'] = np.floor(data['trial_duration_granted'] / 7)

# Last Week Engagement
# Function to populate data from week grouping to last_week feature 
def get_last_week_value(week, w1, w2, w3, w4, w5, w6):
    #week = int(week)
    
    if week > 5  : return w6
    elif week == 5: return w5
    elif week == 4: return w4
    elif week == 3: return w3
    elif week == 2: return w2
    elif week == 1: return w1
    else         : return float(0)

# Engagement for Last Week of Trial
data['last_week_total_video_views'] = data.apply(lambda x: get_last_week_value(x['trial_duration_weeks'],
                                                                               x['w1_total_video_views'],
                                                                               x['w2_total_video_views'],
                                                                               x['w3_total_video_views'],
                                                                               x['w4_total_video_views'],
                                                                               x['w5_total_video_views'],
                                                                               x['w6_total_video_views']), axis=1)


# Last Week / First Week Growth
data['last_first_total_video_views'] = data.apply(lambda x: x['last_week_total_video_views']/x['w1_total_video_views'] - 1 if x['w1_total_video_views'] > 0 else 0, axis=1)
# Network Effect Features

# Prior Trials at School
# group by school ncessch id and start_date, then apply rank minus 1 to find prior trials at school

data['prior_school_trials'] = data.groupby('ncessch')['start_datetime'].rank(method='dense', ascending=True) - 1
# divide the data into numerical ("quan") and categorical ("qual") features
quan = list( data.loc[:,data.dtypes != 'object'].columns.values )
qual = list( data.loc[:,data.dtypes == 'object'].columns.values )
# Find missing values for quantitative and categorical features

#Quantitative
hasNAN = data[quan].isnull().sum()
hasNAN = hasNAN[hasNAN > 0]
hasNAN = hasNAN.sort_values(ascending=False)
print(hasNAN)

print('**'*40)

#Categorical
hasNAN = data[qual].isnull().sum()
hasNAN = hasNAN[hasNAN > 0]
hasNAN = hasNAN.sort_values(ascending=False)
print(hasNAN)
# Drop Columns

# Drop assignments_created_per_teacher (null in 95% of rows)
data = data.drop(['w6_assignments_created_per_teacher', 'w5_assignments_created_per_teacher', 'w4_assignments_created_per_teacher',
                  'w3_assignments_created_per_teacher', 'w2_assignments_created_per_teacher', 'w1_assignments_created_per_teacher',
                  'assignments_created_per_teacher'], axis = 1)

# Drop mzip4, lzip4, mstreet3, lstreet3, mstreet2, lstreet2
data = data.drop(['mzip4', 'lzip4', 'mstreet3', 'lstreet3', 'mstreet2', 'lstreet2'], axis = 1)
# Reassign Naan, Inf, and Irregular Values

# Site Engagement
data['students_participation'].isnull().describe()


# divide the data into numerical ("quan") and categorical ("qual") features
quan = list( engagement_data.loc[:,engagement_data.dtypes != 'object'].columns.values )
qual = list( engagement_data.loc[:,engagement_data.dtypes == 'object'].columns.values )
#Quantitative
hasNAN = engagement_data[quan].isnull().sum()
hasNAN = hasNAN[hasNAN > 0]
hasNAN = hasNAN.sort_values(ascending=False)
print(hasNAN)

print('**'*40)

#Categorical
hasNAN = engagement_data[qual].isnull().sum()
hasNAN = hasNAN[hasNAN > 0]
hasNAN = hasNAN.sort_values(ascending=False)
print(hasNAN)
data[['am_percent','as_percent','bl_percent','wh_percent','hi_percent', 'hp_percent', 'tr_percent', 'member']].describe()
#Subscriber schools have a high percentage of black students on average and a lower percentage of white
data.groupby('is_paid').mean()['bl_percent'].plot.bar()
data.groupby('is_paid').mean()['wh_percent'].plot.bar()
data['totfrl_percent'].replace(np.inf, 0, inplace=True)
data['totfrl_percent'].describe()
totfrl_greater_than_member = data['totfrl'] > data['member']
data[totfrl_greater_than_member][['totfrl', 'member', 'wh', 'bl']]
data['pupfte'] = data['member'] / data['fte']
data['pupfte'].describe()
# Deal with situations when fte or member are negative (unknown)
data[data['member'] < 0][['fte', 'member']].describe()
# Only include trials that ended before 2018
completed_trials = (data['trial_end_date'] <= '2018-01-01')
data = data[completed_trials]

#Check to make sure all 'is_current' values are 0
data.is_current.describe()
sn = pd.DataFrame(data.statename.value_counts())
null_school_info_percent = round(data.sch_name.isnull().sum()/len(data)*100, 1)
school_info_percent = round(data.sch_name.count()/len(data)*100, 1)

print(null_school_info_percent, "% of entries have no school info (null values)")
print(school_info_percent, "% of entries have school info")
data.isnull().sum()
sns.countplot(x='is_paid', data=data)
subscription_rate = round(data.is_paid.mean()*100, 1)
print("Only", subscription_rate, "% of trials converted into subscriptions")
engagement_data = data[['start_date', 'trial_end_date', 'is_paid', 'total_video_views',
                       'teacher_video_views', 'total_assignments_created', 'unique_teachers_creating_assignments',
                       'num_quote_requests', 'active_students', 'active_teachers', 'total_teacher_count',
                       'total_student_count', 'activated_challenge_question_mode', 'num_student_activators',
                       'num_teacher_activators', 'activated_interactive_lyrics_mode', 'activated_quiz_mode',
                       'activated_lesson_plans_mode', 'activated_rapbox_mode', 'activated_handouts_mode',
                       'video_views_per_teacher', 'assignments_created_per_teacher']]
engagement_data.describe()
#Replace null values with 0's
engagement_data = engagement_data.fillna(0)
engagement_data.isnull().sum()
##Total video views without outliers
plt.figure(figsize=(10, 5))
sns.boxplot(engagement_data.total_video_views, showfliers=False)
##Total video views with outliers
sns.boxplot(engagement_data.total_video_views)
#Total video views distribution
sns.kdeplot(engagement_data.total_video_views)
# Number of trials with video views over 50
video_views_over_50 = (engagement_data['total_video_views'] > 500)
over_50 = data[video_views_over_50]['total_video_views'].count()
percent_over_50 = round(over_50/len(engagement_data), 3) * 100
print(percent_over_50, "% of trials had over 50 video views")
#conversions on trials with over 50 views
print(data[video_views_over_50]['is_paid'].sum())
data[video_views_over_50]['is_paid'].count()
# Convert start_date from string to datetime
engagement_data['start_datetime']    = pd.to_datetime(engagement_data['start_date'])
# Month
engagement_data['start_month']       = engagement_data['start_datetime'].dt.month
# Year
engagement_data['start_year']        = engagement_data['start_datetime'].dt.year
# Day of the Week
engagement_data['start_day_of_week'] = engagement_data['start_datetime'].dt.dayofweek
# Year-Month
engagement_data['start_year_month']  = engagement_data['start_datetime'].dt.to_period('m')

# Do the same for the overall data DataFrame
data['start_datetime']    = pd.to_datetime(data['start_date'])
# Month
data['start_month']       = data['start_datetime'].dt.month
# Year
data['start_year']        = data['start_datetime'].dt.year
# Day of the Week
data['start_day_of_week'] = data['start_datetime'].dt.dayofweek
# Year-Month
data['start_year_month']  = data['start_datetime'].dt.to_period('m')

active_teacher_group = engagement_data.groupby('active_teachers', as_index=False).mean()[['is_paid']]
active_teacher_group.plot(kind='line')
video_views_group = engagement_data.groupby('total_video_views', as_index=False).mean()[['is_paid']]
video_views_group.plot(kind='line', xlim=(0,200))
teacher_video_views_group = engagement_data.groupby('teacher_video_views', as_index=False).mean()[['is_paid']]
teacher_video_views_group.plot(kind='line', xlim=(0,200), ylim=(0,0.4))
bins = [-1, .5, 5, 10, 15, 25, 50, 75, 100, np.inf]
engagement_data['total_video_views_bin']= pd.cut(engagement_data['total_video_views'], bins, labels=False)
engagement_data['total_video_views_bin'].max()
bin_video_views_group = engagement_data.groupby('total_video_views_bin', as_index=False).mean()[['is_paid']]

bin_video_views_group.plot(kind='bar')
engagement_data['total_video_views_bin'].describe()
assignments_created_group = engagement_data.groupby('total_assignments_created', as_index=False).mean()[['is_paid']]
assignments_created_group.plot(kind='line', xlim=(0,25))
teachers_make_assignment_group = engagement_data.groupby('unique_teachers_creating_assignments', as_index=False).mean()[['is_paid']]
teachers_make_assignment_group.plot(kind='line', xlim=(0,12))
quotes_group = engagement_data.groupby('num_quote_requests', as_index=False).mean()[['is_paid']]
quotes_group.plot(kind='line', xlim=(0,17))
question_mode_group = engagement_data.groupby('activated_challenge_question_mode', as_index=False).mean()[['is_paid']]
question_mode_group.plot(kind='line', xlim=(0,100))
active_students_group = engagement_data.groupby('active_students', as_index=False).mean()[['is_paid']]
active_students_group.plot(kind='line', xlim=(0,100))
engagement_data.hist(figsize=(10,10))
engagement_data.groupby('is_paid').hist(figsize=(10,10))
# Trials by month_year
start_year_month = engagement_data.groupby('start_year_month').count()[['is_paid']]
start_year_month.sort_index(ascending=True).plot(kind='line')
# Subscriptions by month_year
subscriber_data = engagement_data['is_paid'] == 1
subscriber_data = engagement_data[subscriber_data]
year_month_subscriptions = subscriber_data.groupby('start_year_month').count()[['is_paid']]
year_month_subscriptions.sort_index(ascending=True).plot(kind='line')
# Conversion Rate by month_year
year_month_cr = engagement_data.groupby('start_year_month').mean()[['is_paid']]
year_month_cr.sort_index(ascending=True).plot(kind='line')
# Conversion Rate by month
month_cr = engagement_data.groupby('start_month').mean()[['is_paid']]
month_cr.sort_index(ascending=True).plot(kind='bar')
# Conversion Rate by day of the week
day_of_week_cr = engagement_data.groupby('start_day_of_week').mean()[['is_paid']]
day_of_week_cr.sort_index(ascending=True).plot(kind='bar')
# CR by year
year_cr = engagement_data[(engagement_data['start_year'] > 2012)]
year_cr = year_cr.groupby('start_year').mean()[['is_paid']]
year_cr.sort_index(ascending=True).plot(kind='bar')
# CR by month 2
exclude_11_12 = engagement_data[(engagement_data['start_year'] > 2012)]
month_cr = exclude_11_12.groupby(['start_year','start_month']).mean()[['is_paid']]
month_cr.head()
engagement_data.info()
engagement_data['active_teachers'] = engagement_data['active_teachers'].apply(lambda x: x if x < 20 else 20)
engagement_data['active_teachers'].describe()
engagement_data['active_teachers'].hist(bins=20)

at_impact = engagement_data.groupby('active_teachers', as_index=False).mean()[['is_paid']]
at_impact.plot(kind='line', xlim=(0,20))
engagement_data.info()
# students per teacher =  activate_students / active_teachers
engagement_data['students_per_teacher'] = engagement_data['active_students']/engagement_data['active_teachers']
engagement_data['students_per_teacher'].fillna(0)

# Student Participation Rate
# active students / total students
engagement_data['students_participation'] = engagement_data['active_students']/engagement_data['total_student_count']
engagement_data[engagement_data['total_student_count'] > 1].describe()

# Teacher Participation Rate
# active teachers / total teachers
engagement_data['teacher_participation'] = engagement_data['active_teachers']/engagement_data['total_teacher_count']
#Side Activities (Quiz, lessons, rapbox, handout)
engagement_data.groupby('start_year').sum()
# Convert end_date to datetime
data['end_datetime'] = pd.to_datetime(data['end_date'])
#data['trial_duration_granted']
#data['trial_duration_granted'] = data['trial_duration_granted'].fillna(value=(data.end_datetime - data.start_datetime))
data.groupby('start_year_month')['trial_duration_granted'].mean().plot(kind='line')
# Convert end_date to datetime
data['end_datetime'] = pd.to_datetime(data['end_date'])

## Trial duration / 7 to get the number of trial weeks, rounded down to last full week
data['trial_duration_weeks'] = np.floor(data['trial_duration_granted'] / 7) 
    
## Make that week our number
## If trial is longer than the number of weeks, use six weeks / last week
# Convert end_date to datetime
data['end_datetime'] = pd.to_datetime(data['end_date'])

## Trial duration / 7 to get the number of trial weeks, rounded down to last full week
data['trial_duration_weeks'] = np.floor(data['trial_duration_granted'] / 7)


def get_last_week_value(week, w1, w2, w3, w4, w5, w6):
    #week = int(week)
    
    if week > 5  : return w6
    elif week == 5: return w5
    elif week == 4: return w4
    elif week == 3: return w3
    elif week == 2: return w2
    elif week == 1: return w1
    else         : return float(0)

data['last_week_total_video_views'] = data.apply(lambda x: get_last_week_value(x['trial_duration_weeks'],
                                                                               x['w1_total_video_views'],
                                                                               x['w2_total_video_views'],
                                                                               x['w3_total_video_views'],
                                                                               x['w4_total_video_views'],
                                                                               x['w5_total_video_views'],
                                                                               x['w6_total_video_views']), axis=1)


data['last_first_total_video_views'] = data.apply(lambda x: x['last_week_total_video_views']/x['w1_total_video_views'] - 1 if x['w1_total_video_views'] > 0 else 0, axis=1)
# Sets to 0 if 1w was 0 (to fix dividing by 0). However a better solution is possible.
data['last_first_total_video_views'] = data.apply(lambda x: x['last_week_total_video_views']/x['w1_total_video_views'] - 1 if x['w1_total_video_views'] > 0 else 0, axis=1)
data['last_first_total_video_views'].describe()
#data.groupby('ncessch').count().sort_values('is_paid', ascending=False)

# Prior Trials at School
# sort by id, start_date ascending then rank - 1

data['prior_school_trials'] = data.groupby('ncessch')['start_datetime'].rank(method='dense', ascending=True) - 1
data['prior_school_trials'].unique()
pd.set_option('float_format', '{:f}'.format)
data[data['prior_school_trials'] == 3]['ncessch'].head()
data[data['ncessch'] == 348036402991].sort_values('start_datetime')[['name','start_date', 'prior_school_trials']]
data.groupby('prior_school_trials').mean()['is_paid'].plot(kind='bar')
data['prior_school_trials'].hist(bins=11)

