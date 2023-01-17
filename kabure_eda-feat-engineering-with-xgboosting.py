import subprocess
print('# Line count:')
for file in ['train.csv', 'test.csv', 'train_sample.csv']:
    lines = subprocess.run(['wc', '-l', '../input/{}'.format(file)], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(lines, end='', flush=True)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
df_talk = pd.read_csv("../input/train.csv", nrows=2500000,parse_dates=['click_time'])
df_talk.info()
df_talk.nunique()
df_talk.head()
def datetime_to_deltas(series, delta=np.timedelta64(1, 's')):
    t0 = series.min()
    return ((series-t0)/delta).astype(np.int32)

df_talk['sec'] = datetime_to_deltas(df_talk.click_time)
df_talk['day'] = df_talk['click_time'].dt.day.astype('uint8')
df_talk['hour'] = df_talk['click_time'].dt.hour.astype('uint8')
df_talk['minute'] = df_talk['click_time'].dt.minute.astype('uint8')
df_talk['second'] = df_talk['click_time'].dt.second.astype('uint8')
df_talk['week'] = df_talk['click_time'].dt.dayofweek.astype('uint8')

df_talk.head()
print("The proportion of downloaded over just click: ")
print(round((df_talk.is_attributed.value_counts() / len(df_talk.is_attributed) * 100),2))
print(" ")
print("Downloaded over just clicks description: ")
print(df_talk.is_attributed.value_counts())

plt.figure(figsize=(8, 5))
sns.set(font_scale=1.2)
mean = (df_talk.is_attributed.values == 1).mean()

ax = sns.barplot(['Fraudulent (1)', 'Not Fradulent (0)'], [mean, 1-mean])
ax.set_xlabel('Target Value', fontsize=15) 
ax.set_ylabel('Probability', fontsize=15)
ax.set_title('Target value distribution', fontsize=20)

for p, uniq in zip(ax.patches, [mean, 1-mean]):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height+0.01,
            '{}%'.format(round(uniq * 100, 2)),
            ha="center") 
ip_frequency_downloaded = df_talk[df_talk['is_attributed'] == 1]['ip'].value_counts()[:20]
ip_frequency_click = df_talk[df_talk['is_attributed'] == 0]['ip'].value_counts()[:20]

plt.figure(figsize=(16,10))
plt.subplot(2,1,1)
g = sns.barplot(ip_frequency_downloaded.index, ip_frequency_downloaded.values, color='blue')
g.set_title("TOP 20 IP's where the click come from was downloaded",fontsize=20)
g.set_xlabel('Most frequents IPs',fontsize=16)
g.set_ylabel('Count',fontsize=16)

plt.subplot(2,1,2)
g1 = sns.barplot(ip_frequency_click.index, ip_frequency_click.values, color='blue')
g1.set_title("TOP 20 IP's where the click come from was NOT downloaded",fontsize=20)
g1.set_xlabel('Most frequents IPs',fontsize=16)
g1.set_ylabel('Count',fontsize=16)

plt.subplots_adjust(wspace = 0.1, hspace = 0.4,top = 0.9)

plt.show()
app_frequency_downloaded = df_talk[df_talk['is_attributed'] == 1]['app'].value_counts()[:20]
app_frequency_click = df_talk[df_talk['is_attributed'] == 0]['app'].value_counts()[:20]

plt.figure(figsize=(16,10))
plt.subplot(2,1,1)
g = sns.barplot(app_frequency_downloaded.index, app_frequency_downloaded.values,
                palette='husl')
g.set_title("TOP 20 APP where the click come from and downloaded",fontsize=20)
g.set_xlabel('Most frequents APP ID',fontsize=16)
g.set_ylabel('Count',fontsize=16)

plt.subplot(2,1,2)
g1 = sns.barplot(app_frequency_click.index, app_frequency_click.values,
                palette='husl')
g1.set_title("TOP 20 APP where the click come from NOT downloaded",fontsize=20)
g1.set_xlabel('Most frequents APP ID',fontsize=16)
g1.set_ylabel('Count',fontsize=16)

plt.subplots_adjust(wspace = 0.1, hspace = 0.4,top = 0.9)

plt.show()
print("App percentual distribuition description: ")
print(round(df_talk[df_talk['is_attributed'] == 1]['app'].value_counts()[:5] \
            / len(df_talk[df_talk['is_attributed'] == 1]) * 100),2)
channel_frequency_downloaded = df_talk[df_talk['is_attributed'] == 1]['channel'].value_counts()[:20]
channel_frequency_click = df_talk[df_talk['is_attributed'] == 0]['channel'].value_counts()[:20]

plt.figure(figsize=(16,10))

plt.subplot(2,1,1)
g = sns.barplot(channel_frequency_downloaded.index, channel_frequency_downloaded.values, \
                palette='husl')
g.set_title("TOP 20 channels with download Count",fontsize=20)
g.set_xlabel('Most frequents Channels ID',fontsize=16)
g.set_ylabel('Count',fontsize=16)

plt.subplot(2,1,2)
g1 = sns.barplot(channel_frequency_click.index, channel_frequency_click.values,\
                 palette='husl')
g1.set_title("TOP 20 channels clicks Count",fontsize=20)
g1.set_xlabel('Most frequents Channels ID',fontsize=16)
g1.set_ylabel('Count',fontsize=16)

plt.subplots_adjust(wspace = 0.1, hspace = 0.4,top = 0.9)

plt.show()
print("Channel percentual distribuition description: ")
print(round(df_talk[df_talk['is_attributed'] == 1]['channel'].value_counts()[:5] \
            / len(df_talk[df_talk['is_attributed'] == 1]) * 100),2)
device_frequency_downloaded = df_talk[df_talk['is_attributed'] == 1]['device'].value_counts()[:20]
device_frequency_click = df_talk[df_talk['is_attributed'] == 0]['device'].value_counts()[:20]

plt.figure(figsize=(16,10))
plt.subplot(2,1,1)
g = sns.barplot(device_frequency_downloaded.index, device_frequency_downloaded.values,
                palette='husl')
g.set_title("TOP 20 devices with download - Count",fontsize=20)
g.set_xlabel('Most frequents Devices ID',fontsize=16)
g.set_ylabel('Count',fontsize=16)

plt.subplot(2,1,2)
g1 = sns.barplot(device_frequency_click.index, device_frequency_click.values,
                palette='husl')
g1.set_title("TOP 20 devices with download - Count",fontsize=20)
g1.set_xlabel('Most frequents Devices ID',fontsize=16)
g1.set_ylabel('Count',fontsize=16)

plt.subplots_adjust(wspace = 0.1, hspace = 0.4,top = 0.9)

plt.show()
print("Device percentual distribuition: ")
print(round(df_talk[df_talk['is_attributed'] == 1]['device'].value_counts()[:5] \
            / len(df_talk[df_talk['is_attributed'] == 1]) * 100),2)
os_frequency_downloaded = df_talk[df_talk['is_attributed'] == 1]['os'].value_counts()[:20]
os_frequency_click = df_talk[df_talk['is_attributed'] == 0]['os'].value_counts()[:20]

plt.figure(figsize=(16,10))
plt.subplot(2,1,1)
g = sns.barplot(os_frequency_downloaded.index, os_frequency_downloaded.values,
                palette='husl')
g.set_title("TOP 20 OS with download - Count",fontsize=20)
g.set_xlabel("Most frequents OS's ID",fontsize=16)
g.set_ylabel('Count',fontsize=16)

plt.subplot(2,1,2)
g1 = sns.barplot(os_frequency_downloaded.index, os_frequency_downloaded.values,
                palette='husl')
g1.set_title("TOP 20 OS with download - Count",fontsize=20)
g1.set_xlabel("Most frequents OS's ID",fontsize=16)
g1.set_ylabel('Count',fontsize=16)

plt.subplots_adjust(wspace = 0.1, hspace = 0.4,top = 0.9)

plt.show()
print("Device percentual distribuition: ")
print(round(df_talk[df_talk['is_attributed'] == 1]['os'].value_counts()[:5] \
            / len(df_talk[df_talk['is_attributed'] == 1]) * 100),2)
hour_frequency_downloaded = df_talk[df_talk['is_attributed'] == 1]['hour'].value_counts()
hour_frequency_click = df_talk[df_talk['is_attributed'] == 0]['hour'].value_counts()

plt.figure(figsize=(16,10))
plt.subplot(2,1,1)
g = sns.barplot(hour_frequency_downloaded.index, hour_frequency_downloaded.values,
                palette='husl')
g.set_title("Downloads Count by Hour",fontsize=20)
g.set_xlabel("Hour Download distribuition",fontsize=16)
g.set_ylabel('Count',fontsize=16)

plt.subplot(2,1,2)
g1 = sns.barplot(hour_frequency_click.index, hour_frequency_click.values,
                palette='husl')
g1.set_title("Clicks Count by Hour",fontsize=20)
g1.set_xlabel("Hour Click distribuition",fontsize=16)
g1.set_ylabel('Count',fontsize=16)

plt.subplots_adjust(wspace = 0.1, hspace = 0.4,top = 0.9)

plt.show()
df_talk['click_nanosecs'] = (df_talk['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
df_talk['next_click'] = (df_talk.groupby(['ip', 'app', 'device', 'os']).click_nanosecs.shift(-1) - df_talk.click_nanosecs).astype(np.float32)
df_talk['next_click'].fillna((df_talk['next_click'].mean()), inplace=True)

minute_frequency_downloaded = df_talk[df_talk['is_attributed'] == 1]['minute'].value_counts()
minute_frequency_click = df_talk[df_talk['is_attributed'] == 0]['minute'].value_counts()

plt.figure(figsize=(16,10))
plt.subplot(2,1,1)
g = sns.barplot(minute_frequency_downloaded.index, minute_frequency_downloaded.values,
                palette='husl')
g.set_title("Downloads Count by Minute",fontsize=20)
g.set_xlabel("Minute Download distribuition",fontsize=16)
g.set_ylabel('Count',fontsize=16)

plt.subplot(2,1,2)
g1 = sns.barplot(minute_frequency_click.index, minute_frequency_click.values,
                palette='husl')
g1.set_title("Clicks Count by Minute",fontsize=20)
g1.set_xlabel("Minute Click distribuition",fontsize=16)
g1.set_ylabel('Count',fontsize=16)

plt.subplots_adjust(wspace = 0.1, hspace = 0.4,top = 0.9)

plt.show()
second_frequency_downloaded = df_talk[df_talk['is_attributed'] == 1]['second'].value_counts()
second_frequency_click = df_talk[df_talk['is_attributed'] == 0]['second'].value_counts()

plt.figure(figsize=(16,10))
plt.subplot(2,1,1)
g = sns.barplot(second_frequency_downloaded.index, second_frequency_downloaded.values,
                palette='husl')
g.set_title("Downloads Count by Hour",fontsize=20)
g.set_xlabel("Second Download distribuition",fontsize=16)
g.set_ylabel('Count',fontsize=16)

plt.subplot(2,1,2)
g1 = sns.barplot(second_frequency_click.index, second_frequency_click.values,
                palette='husl')
g1.set_title("Clicks Count by Hour",fontsize=20)
g1.set_xlabel("Second Click distribuition",fontsize=16)
g1.set_ylabel('Count',fontsize=16)

plt.subplots_adjust(wspace = 0.1, hspace = 0.4,top = 0.9)

plt.show()
import gc
#Define all the groupby transformations
GROUPBY_AGGREGATIONS = [
    
    # V1 - GroupBy Features #
    #########################    
    # Variance in day, for ip-app-channel
    {'groupby': ['ip','app','channel'], 'select': 'day', 'agg': 'var'},
    # Variance in hour, for ip-app-os
    {'groupby': ['ip','app','os'], 'select': 'hour', 'agg': 'var'},
    # Variance in hour, for ip-day-channel
    {'groupby': ['ip','day','channel'], 'select': 'hour', 'agg': 'var'},
    # Count, for ip-day-hour
    {'groupby': ['ip','day','hour'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app
    {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'count'},        
    # Count, for ip-app-os
    {'groupby': ['ip', 'app', 'os'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app-day-hour
    {'groupby': ['ip','app','day','hour'], 'select': 'channel', 'agg': 'count'},
    # Mean hour, for ip-app-channel
    {'groupby': ['ip','app','channel'], 'select': 'hour', 'agg': 'mean'}, 
    
    # V2 - GroupBy Features #
    #########################
    # Average clicks on app by distinct users; is it an app they return to?
    {'groupby': ['app'], 
     'select': 'ip', 
     'agg': lambda x: float(len(x)) / len(x.unique()), 
     'agg_name': 'AvgViewPerDistinct'
    },
    # How popular is the app or channel?
    {'groupby': ['app'], 'select': 'channel', 'agg': 'count'},
    {'groupby': ['channel'], 'select': 'app', 'agg': 'count'},
    
    # V3 - GroupBy Features                                              #
    # https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977 #
    ###################################################################### 
    {'groupby': ['ip'], 'select': 'channel', 'agg': 'nunique'}, 
    {'groupby': ['ip'], 'select': 'app', 'agg': 'nunique'}, 
    {'groupby': ['ip','day'], 'select': 'hour', 'agg': 'nunique'}, 
    {'groupby': ['ip','app'], 'select': 'os', 'agg': 'nunique'}, 
    {'groupby': ['ip'], 'select': 'device', 'agg': 'nunique'}, 
    {'groupby': ['app'], 'select': 'channel', 'agg': 'nunique'}, 
    {'groupby': ['ip', 'device', 'os'], 'select': 'app', 'agg': 'nunique'}, 
    {'groupby': ['ip','device','os'], 'select': 'app', 'agg': 'cumcount'}, 
    {'groupby': ['ip'], 'select': 'app', 'agg': 'cumcount'}, 
    {'groupby': ['ip'], 'select': 'os', 'agg': 'cumcount'}, 
    {'groupby': ['ip','day','channel'], 'select': 'hour', 'agg': 'var'}    
]

# Apply all the groupby transformations
for spec in GROUPBY_AGGREGATIONS:
    
    # Name of the aggregation we're applying
    agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']
    
    # Name of new feature
    new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])
    
    # Info
    print("Grouping by {}, and aggregating {} with {}".format(
        spec['groupby'], spec['select'], agg_name
    ))
    
    # Unique list of features to select
    all_features = list(set(spec['groupby'] + [spec['select']]))
    
    # Perform the groupby
    gp = df_talk[all_features]. \
        groupby(spec['groupby'])[spec['select']]. \
        agg(spec['agg']). \
        reset_index(). \
        rename(index=str, columns={spec['select']: new_feature})
        
    # Merge back to X_total
    if 'cumcount' == spec['agg']:
        df_talk[new_feature] = gp[0].values
    else:
        df_talk = df_talk.merge(gp, on=spec['groupby'], how='left')
        
     # Clear memory
    del gp
    gc.collect()
import xgboost as xgb

# Split into X and y
y = df_talk['is_attributed']
X = df_talk.drop('is_attributed', axis=1).select_dtypes(include=[np.number])

# Create a model
# Params from: https://www.kaggle.com/aharless/swetha-s-xgboost-revised
clf_xgBoost = xgb.XGBClassifier(
    max_depth = 4,
    subsample = 0.8,
    colsample_bytree = 0.7,
    colsample_bylevel = 0.7,
    scale_pos_weight = 9,
    min_child_weight = 0,
    reg_alpha = 4,
    n_jobs = 4, 
    objective = 'binary:logistic'
)
# Fit the models
clf_xgBoost.fit(X, y)
from sklearn import preprocessing

# Get xgBoost importances
feature_importance = {}
for import_type in ['weight', 'gain', 'cover']:
    feature_importance['xgBoost-'+import_type] = clf_xgBoost.get_booster().get_score(importance_type=import_type)
    
# MinMax scale all importances
features = pd.DataFrame(feature_importance).fillna(0)
features = pd.DataFrame(
    preprocessing.MinMaxScaler().fit_transform(features),
    columns=features.columns,
    index=features.index
)

# Create mean column
features['mean'] = features.mean(axis=1)

# Plot the feature importances
features.sort_values('mean').plot(kind='bar', figsize=(16, 6))
plt.show()


