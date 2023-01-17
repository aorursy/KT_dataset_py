import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/train_sessions.csv',
                       index_col='session_id')
test_df = pd.read_csv('../input/test_sessions.csv',
                      index_col='session_id')

# Convert time1, ..., time10 columns to datetime type
times = ['time%s' % i for i in range(1, 11)]
train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

# Sort the data by time
train_df = train_df.sort_values(by='time1')

# Look at the first rows of the training set
train_df.head()
sites = ['site%s' % i for i in range(1, 11)]
train_df[sites].fillna(0).astype('int').to_csv('train_sessions_text.txt', 
                                               sep=' ', 
                       index=None, header=None)
test_df[sites].fillna(0).astype('int').to_csv('test_sessions_text.txt', 
                                              sep=' ', 
                       index=None, header=None)
!head -1 train_sessions_text.txt
# win equivalent
# !powershell -ExecutionPolicy Bypass "Get-Content .\train_sessions_text.txt -TotalCount 0.001kb" ;
# Load websites dictionary
with open("../input/site_dic.pkl", "rb") as input_file:
    site_dict = pickle.load(input_file)

# Create dataframe for the dictionary
sites_dict = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), columns=['site'])
print(u'Websites total:', sites_dict.shape[0])
sites_dict.head()
top_sites = pd.Series(train_df[sites].fillna(0).values.flatten()
                     ).value_counts().sort_values(ascending=False)
top_sites.head()
alice_sites = pd.Series(train_df[train_df.target==1][sites].fillna(0).values.flatten()
                           ).value_counts().sort_values(ascending=False)
alice_sites.head()
others_sites = pd.Series(train_df[train_df.target==0][sites].fillna(0).values.flatten()
                           ).value_counts().sort_values(ascending=False)
others_sites.head()
alice_sites.shape, others_sites.shape, top_sites.shape
#Training data sites percentage
top_sites.shape[0]/sites_dict.shape[0]*100
#Alice's sites percentage
alice_sites.shape[0]/sites_dict.shape[0]*100
#Others' sites percentage
others_sites.shape[0]/sites_dict.shape[0]*100
sns.barplot(x = ['Alice', 'Alice+Others', 'Only Alice'], y=[alice_sites.shape[0], 
                                                            alice_sites.shape[0]+others_sites.shape[0]-top_sites.shape[0],
                                                            top_sites.shape[0] - others_sites.shape[0]])
plt.title('Number of websites comparison in Alice''s URL set');
plt.ylabel('# of websites');
train_df_sites = train_df[sites].fillna(0).astype(int)
train_df_sites.shape
top_sites_df = sites_dict.ix[top_sites.index]
top_sites_df['freq'] = top_sites
top_sites_df.head()
alice_sites_df = sites_dict.ix[alice_sites.index]
alice_sites_df['freq'] = alice_sites
alice_sites_df.head()
others_sites_df = sites_dict.ix[others_sites.index]
others_sites_df['freq'] = others_sites
others_sites_df.head()
alice_unique_sites_index = top_sites_df.index.difference(others_sites_df.index)
alice_unique_sites_index
alice_unique_sites_df = alice_sites.loc[alice_unique_sites_index]
alice_unique_sites_df = pd.DataFrame(alice_unique_sites_df, columns=['freq'])
alice_unique_sites_df['site'] = sites_dict.ix[alice_unique_sites_df.index]
alice_unique_sites_df.head()
alice_unique_sorted_urls = alice_unique_sites_df.sort_values(by='freq',ascending=False)
alice_unique_sorted_urls.head(5)
df = train_df
hour = df['time1'].apply(lambda ts: ts.hour)
df['hour'] = df['time1'].apply(lambda ts: ts.hour)
df['morning'] = ((hour >= 7) & (hour <= 11)).astype('int')
df['noon'] = ((hour >= 12) & (hour <= 13)).astype('int')
df['afternoon'] = ((hour >= 14) & (hour <= 18)).astype('int')
df['day'] = ((hour >= 12) & (hour <= 18)).astype('int')
df['evening'] = ((hour >= 19) & (hour <= 23)).astype('int')
df['late_evening'] = ((hour >= 21) & (hour <= 23)).astype('int')
df['night'] = ((hour >= 0) & (hour <= 6)).astype('int')
df['early_night'] = ((hour >= 0) & (hour <= 2)).astype('int')
df['late_night'] = ((hour >= 3) & (hour <= 6)).astype('int')
weekday = df['time1'].apply(lambda ts: ts.dayofweek)
df['weekday'] = df['time1'].apply(lambda ts: ts.dayofweek)
df['weekend'] = ((weekday >= 5) & (weekday <= 6)).astype('int')
df['weekdays'] = (weekday <= 4).astype('int')
df['years'] = df['time1'].apply(lambda ts: ts.year)
df['weeks'] = df['time1'].apply(lambda ts: 100 * ts.year + ts.week)

# and soo on...
df_uniques = pd.melt(frame=df[df['target']==0], value_vars=['hour'])

df_uniques = pd.DataFrame(df_uniques.groupby(['variable', 
                                              'value'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()

sns.factorplot(x='variable', y='count', hue='value', 
               data=df_uniques, kind='bar', size=6);
df_uniques = pd.melt(frame=df[df['target']==1], value_vars=['hour'])

df_uniques = pd.DataFrame(df_uniques.groupby(['variable', 
                                              'value'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()

sns.factorplot(x='variable', y='count', hue='value', 
               data=df_uniques, kind='bar', size=6);
def snsplot(df, feature):
    #Intruder's data
    df_uniques = pd.melt(frame=df[df['target']==0], value_vars=[feature])
    df_uniques = pd.DataFrame(df_uniques.groupby(['variable', 
                                              'value'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()

    sns.factorplot(x='variable', y='count', hue='value', 
               data=df_uniques, kind='bar', size=6);
    plt.title('Intruder')
    
    # Now plot Alice's data
    df_uniques = pd.melt(frame=df[df['target']==1], value_vars=[feature])
    df_uniques = pd.DataFrame(df_uniques.groupby(['variable', 
                                              'value'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()

    sns.factorplot(x='variable', y='count', hue='value', 
               data=df_uniques, kind='bar', size=6);
    plt.title('Alice')
snsplot(df,'weekday')
snsplot(df,'weekdays')
snsplot(df,'years')
snsplot(df,'weeks')