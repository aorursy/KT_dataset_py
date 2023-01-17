# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from IPython.display import display

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = np.random.seed(0)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
'''
Importing Tables and checking memory used
'''

activities = pd.read_csv('/kaggle/input/coding-doubt-activity/activities.csv', parse_dates = ['created_at'])
doubts = pd.read_csv('/kaggle/input/coding-doubt-activity/doubts.csv', parse_dates = ['created_at'])

display(activities)
print('Activities Set Memory Usage = {:.2f} MB'.format(activities.memory_usage().sum() / 1024**2))
print('----------------------------------------')
display(doubts)
print('doubts Set Memory Usage = {:.2f} MB'.format(doubts.memory_usage().sum() / 1024**2))
def draw_missing_data_table(df):
    '''
    Docstring: Returns a datarframe with percent of missing/nan values per feature/column
    
    Parameters:
    ------------
    df: dataframe object
    
    Returns:
    ------------
    Dataframe containing missing value information
    '''
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
draw_missing_data_table(activities)
draw_missing_data_table(doubts)
doubts.user_rating.fillna('NR',inplace=True)
# getting the unique activity_types for reference
activities.activity_type.unique()
def is_student(x):
    if (x['activity_type'] in ['rate', 'review_not_resolved', 'not_pending_action_done', 'not_pending_info_submit','resolve']):
        return '1'
    else:
        return '0'

activities['is_student_id'] = activities.apply(lambda x:is_student(x),axis=1)
activities.is_student_id = activities.is_student_id.astype(int)
def is_TA(x):
    if (x['activity_type'] in ['activate','reject','pending_action_required','pending_information_required',
                               'review_resolution','assign','accept','reactivate','escalate']):
        return '1'
    else:
        return '0'

activities['is_TA_id'] = activities.apply(lambda x:is_TA(x),axis=1)
activities.is_TA_id = activities.is_TA_id.astype(int)
def is_sys(x):
    if (x['activity_type'] in ['unassign', 'inactivate', 'pending_sms', 'pending_reminder_first', 'pending_reminder_second', 'dead']):
        return '1'
    else:
        return '0'

activities['is_sys_id'] = activities.apply(lambda x:is_sys(x),axis=1)
activities.is_sys_id = activities.is_sys_id.astype(int)
activities
# id in doubts table (doubt_id) is unique (primary key)
doubts
activities.loc[activities.activity_type=='available']
'''
Standardizing Time to minutes
'''

def convert_timedelta(duration):
    days, seconds = duration.days, duration.seconds

    minutes = (seconds % 3600) / 60
    return minutes
activities.loc[activities.doubt_id==248205]
# pick one id from doubts table (doubt_id) and extract corresponding created in activities table

dlc = []
drt = []
TA_id = []
with tqdm(total=len(doubts)) as pbar:
    for index, row in doubts.iterrows():
        doubt_id = row['id']
        
        # finding when was the doubt attained its first "assign" state by sorting through all assign states and picking the earliest
        try:
            # Doubt_life_cycle
            assign_time = activities.loc[(activities.doubt_id==doubt_id) & 
                                         (activities.activity_type=='assign'),
                                         ('created_at')].sort_values().iloc[0]

            resolution_time = row['created_at']

            dlc.append(convert_timedelta(resolution_time-assign_time))
            
        except Exception as e:
            #print("Exception Occurred:",e)
            dlc.append(1)
            
        try:
            # DRT
            t = activities.loc[(activities.doubt_id==doubt_id) & 
                                (activities.activity_type=='assign') &
                                (activities.is_TA_id==1),
                                ('created_at','user_id')].sort_values(by='created_at').iloc[-1]
            
            assign_time = t['created_at']
            resolution_time = row['created_at']
            
            TA_id.append(t['user_id'])
            drt.append(convert_timedelta(resolution_time-assign_time))
            
        except Exception as e:
            #print("Exception Occurred:",e)
            drt.append(1)
            TA_id.append('not_assigned/resolve')
            
        pbar.update(1)
        
doubts['Doubt_resolution_time(min)'] = drt
doubts['Doubt_life_cycle(min)'] = dlc
doubts['TA_id'] = TA_id
doubts
fig, ax = plt.subplots(figsize=(15, 5))

x =doubts.groupby('TA_id', as_index=False)['Doubt_resolution_time(min)'].mean().sort_values(by='Doubt_resolution_time(min)')

print('q1=',np.percentile(x['Doubt_resolution_time(min)'], 25))
print('q3=',np.percentile(x['Doubt_resolution_time(min)'], 75))
print("IQR:",np.subtract(*np.percentile(x['Doubt_resolution_time(min)'].values, [75, 25])))

sns.distplot(x['Doubt_resolution_time(min)'].values,ax=ax);
ax.set_title('Sampling Distribution of mean Doubt Resolution by each TA', fontsize=15)
df = doubts.groupby('TA_id', as_index=False)['Doubt_resolution_time(min)'].mean()
fig, ax = plt.subplots(figsize=(15, 5))

TOTAL_NO_OF_PROBLEMS_SOLVED_BY_EACH_TA = doubts.loc[doubts.TA_id!='not_assigned/resolve'].TA_id.value_counts()

print("Total Number of Problems solved by TAs  :",TOTAL_NO_OF_PROBLEMS_SOLVED_BY_EACH_TA.sum())
print("Average Number of Problems solved by TAs:",int(TOTAL_NO_OF_PROBLEMS_SOLVED_BY_EACH_TA.mean()))
sns.distplot(TOTAL_NO_OF_PROBLEMS_SOLVED_BY_EACH_TA.values,ax=ax);
ax.set_title("Distribution of Number of Problems solved by TA's", fontsize=15)
ax.set_xlabel('No of Problems Solved', size=15, labelpad=20)

t = doubts.loc[doubts.user_rating!='NR']  #NR won't be considered for the metric
t.user_rating = t.user_rating.astype(float)
RM = t.groupby('TA_id').agg({'TA_id':'count', 
                            'Doubt_resolution_time(min)':'mean',
                            'user_rating':'mean'
                             })
RM.columns = ['No_of_doubts_cleared','DRT','Mean_user_rating']
RM.drop(['not_assigned/resolve'],inplace=True)
RM['Rank_Metric'] = RM.No_of_doubts_cleared*RM.Mean_user_rating/RM.DRT
display(RM)
top_performers = RM.sort_values(by='Rank_Metric',ascending=False).iloc[:10]

display(top_performers)

fig, ax = plt.subplots(figsize=(15, 8))
sns.barplot(x=top_performers.index,y='Rank_Metric', data=top_performers,ax=ax,capsize=.2)
ax.set_title('Top 10 Performers', fontsize=15)
fig, ax1 = plt.subplots(figsize=(25, 6))
sns.barplot(x="content_type",
            y='Doubt_life_cycle(min)', 
            data= doubts.groupby('content_type', as_index=False)['Doubt_life_cycle(min)'].mean(),
            ax=ax1,
            capsize=.2)
ax1.set_title('Average Time taken by doubt from each Content_type (red horizontal line denotes mean of DLC)', fontsize=15)

ax2 = ax1.twiny()
ax2 = plt.axhline(y=doubts['Doubt_life_cycle(min)'].mean(),color='red')
plt.show()
fig, ax1 = plt.subplots(figsize=(25, 6))

sns.barplot(x="course_id",
            y='Doubt_life_cycle(min)', 
            data= doubts.groupby('course_id', as_index=False)['Doubt_life_cycle(min)'].mean(),
            ax=ax1,
            capsize=.2)
ax1.set_title('Average Time taken by doubt from each Course', fontsize=15)

ax2 = ax1.twiny()
ax2 = plt.axhline(y=doubts['Doubt_life_cycle(min)'].mean(),color='red')
# which content_type recieved the most doubt requests (and were completed)
fig, ax = plt.subplots(figsize=(25, 6))
sns.barplot(x=doubts.content_type.value_counts().index,y=doubts.content_type.value_counts(),ax=ax,capsize=.2)
ax.set_title('Content Type by recieving Number of doubts request', fontsize=15)
ax.set_xlabel('Content Type',fontsize=15)
ax.set_ylabel('Number of Requests',fontsize=15)
pd.set_option('display.max_rows', 68)
display(doubts.pivot_table(index=['course_id','content_type'],
                           values=['Doubt_life_cycle(min)','id'],
                           aggfunc={'Doubt_life_cycle(min)':'mean', 'id':'size'}))
pd.set_option('display.max_rows', 10)
pt = pd.DataFrame(doubts.groupby(['course_id','content_type'], as_index=False)\
                 .agg({'Doubt_life_cycle(min)':'mean', 'id':'size'}) \
                 .rename(columns={'Doubt_life_cycle(min)':'DLC','id':'Doubts_solved'}))
pt
content_types = pt.content_type.unique()
fig, ax = plt.subplots(nrows=int(len(content_types)/2),ncols=2,figsize=(20, 50))
ax=ax.flatten()
dlc_avg = doubts['Doubt_life_cycle(min)'].mean()
for i,ctype in enumerate(content_types):
    t = pt.loc[pt.content_type==ctype]

    g = sns.barplot(x='course_id',y='DLC', data=t, ax=ax[i], capsize=.2)
    
    #for index, row in t.iterrows():
    #    g.text(row.course_id, row.DLC, row.Doubts_solved, color='black', ha="center")
    
    ax[i].set_title(ctype, fontsize=15)
    
    ax2 = ax[i].twiny()
    ax2 = plt.axhline(y=dlc_avg,color='red')
fig, ax = plt.subplots(nrows=int(len(content_types)/2),ncols=2,figsize=(20, 50))
ax=ax.flatten()
req_avg = pt['Doubts_solved'].mean()

for i,ctype in enumerate(content_types):
    t = pt.loc[pt.content_type==ctype]

    g = sns.barplot(x='course_id',y='Doubts_solved', data=t, ax=ax[i], capsize=.2)
    
    #for index, row in t.iterrows():
    #    g.text(row.course_id, row.DLC, row.Doubts_solved, color='black', ha="center")
    
    ax[i].set_title(ctype, fontsize=15)
    
    ax2 = ax[i].twiny()
    ax2 = plt.axhline(y=req_avg,color='red')
DRT_UR = doubts[['Doubt_life_cycle(min)','user_rating']]
DRT_UR = DRT_UR.loc[DRT_UR.user_rating!='NR']
DRT_UR.user_rating = DRT_UR.user_rating.astype(int)
display(DRT_UR)
from scipy import stats
stats.pointbiserialr(DRT_UR.user_rating.values, DRT_UR['Doubt_life_cycle(min)'])
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1.0, 
                        class_weight='balanced', #n_samples / (n_classes * np.bincount(y)) 
                        dual=False, 
                        fit_intercept=True,
                        intercept_scaling=1, 
                        l1_ratio=None, 
                        max_iter=100,
                        multi_class='multinomial', 
                        n_jobs=-1, 
                        penalty='l2',
                        random_state=RANDOM_SEED, 
                        solver='newton-cg', 
                        tol=0.0001, 
                        verbose=0,
                        warm_start=False)
X = DRT_UR['Doubt_life_cycle(min)'].values.reshape(-1,1)
y = DRT_UR.user_rating.values
lr.fit(X,y)
print(lr.coef_)
print(lr.intercept_)
fig, ax = plt.subplots(figsize=(15, 6))
sns.boxplot(x="user_rating", y="Doubt_life_cycle(min)", data=DRT_UR,ax=ax)

ax.set_title('Box Plot to analyze the DLC for every User Rating', fontsize=15)
activities
fig, ax = plt.subplots(figsize=(25, 6))
sns.barplot(x=activities.activity_type.value_counts().index,y=activities.activity_type.value_counts(),ax=ax,capsize=.2)

ax.set_title('Distribution of different activity states', fontsize=15)
ax.set_xlabel('Activity States',fontsize=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.set_ylabel('Count',fontsize=15)
d_id1 = activities.doubt_id.unique()
d_id2 = doubts.id.unique()
np.setdiff1d(d_id1,d_id2,assume_unique=True)