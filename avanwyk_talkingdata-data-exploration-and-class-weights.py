import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

%matplotlib inline
mpl.style.use('seaborn')
sns.set(rc={'figure.figsize':(12,5)});
gfig = plt.figure(figsize=(12,5));
df = pd.read_csv('../input/train.csv', parse_dates=['click_time', 'attributed_time'], nrows=1000000)
categorical = ['ip', 'app', 'device', 'os', 'channel', 'is_attributed']
for c in categorical:
    df[c] = df[c].astype('category')
df.sample(10)
df.describe()
df['is_attributed'].value_counts()
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced', df['is_attributed'].unique(), df['is_attributed'].values)
print("is_attributed==0 weight: {},\nis_attributed==1 weight: {}".format(class_weights[0], class_weights[1]))
fig, axes = plt.subplots(2, 5)
fig.set_figheight(8)
fig.set_figwidth(20)
fig.tight_layout()
attributed = [0, 1]
attributes = ['ip', 'device', 'os', 'app', 'channel']
for attributed in attributed:
    for idx, attr in enumerate(attributes):
        values = df[df.is_attributed == attributed][attr].value_counts().head(20)
        ax = values.plot.bar(ax=axes[attributed][idx])
        ax.set_title(attr)
        if idx == 0:
            if attributed == 0:
                h = ax.set_ylabel('not-attributed', rotation='vertical', size='large')
            else:
                h= ax.set_ylabel('attributed', rotation='vertical', size='large')
plt.subplots_adjust(hspace=0.3)
df['click_h'] = df['click_time'].dt.hour + df['click_time'].dt.minute / 60
df['attributed_h'] = df['attributed_time'].dt.hour + df['attributed_time'].dt.minute / 60
fig, axes = plt.subplots(1, 2)
fig.set_figwidth(20)
ax = df['click_h'].plot.hist(bins=24, ax=axes[0])
xl = ax.set_xlabel('hour')
title = ax.set_title('click_time')

ax = df['attributed_h'].plot.hist(bins=24, ax=axes[1])
xl = ax.set_xlabel('hour')
title = ax.set_title('attributed_time')
df['click_month'] = df['click_time'].dt.month
df['click_day'] = df['click_time'].dt.day
df['click_hour'] = df['click_time'].dt.hour
# convert back to object to avoid an issue with merging with categorical data: https://github.com/pandas-dev/pandas/issues/18646
for c in categorical:
    df[c] = df[c].astype('object')
def create_click_aggregate(frame, name, idxs):
    aggregate = frame.groupby(by=idxs, as_index=False).click_time.count()
    aggregate = aggregate.rename(columns={'click_time': name})
    return frame.merge(aggregate, on=idxs)
def create_attributed_aggregate(frame, name, idxs):
    aggregate = frame[frame['is_attributed'] == 1].groupby(by=idxs, as_index=False).is_attributed.count()
    aggregate = aggregate.rename(columns={'is_attributed': name})
    return frame.merge(aggregate, on=idxs)
df = create_click_aggregate(df, 'total_clicks', ['ip'])
df = create_click_aggregate(df, 'clicks_in_day', ['ip', 'click_month', 'click_day'])
df = create_click_aggregate(df, 'clicks_in_hour', ['ip', 'click_month', 'click_day', 'click_hour'])
df = create_attributed_aggregate(df, 'total_attributions', ['ip'])
df = create_attributed_aggregate(df, 'attributed_in_day', ['ip', 'click_month', 'click_day'])
df = create_attributed_aggregate(df, 'attributed_in_hour', ['ip', 'click_month', 'click_day', 'click_hour'])
fig, axes = plt.subplots(3, 1)
fig.set_figheight(8)
fig.set_figwidth(20)
fig.tight_layout()

time_aggregates = [('total_clicks', 'total_attributions'), ('clicks_in_day', 'attributed_in_day'), ('clicks_in_hour', 'attributed_in_hour')]
row = 0
for time_aggregate in time_aggregates:
    ax = df[['ip', time_aggregate[0], time_aggregate[1]]].drop_duplicates().sort_values(time_aggregate[0], ascending=False).head(20).set_index('ip').plot.bar(ax=axes[row], secondary_y=time_aggregate[1])
    if row == 0:
        ax.set_title('Non-attributed')
    row+=1
def unique_values_by_ip(frame, value):
    n_values_by_ip = frame.groupby(by='ip')[value].nunique()
    frame.set_index('ip', inplace=True)
    frame['n_' + value] = n_values_by_ip
    frame.reset_index(inplace=True)
    return frame
df = unique_values_by_ip(df, 'os')
df = unique_values_by_ip(df, 'app')
df = unique_values_by_ip(df, 'device')
df = unique_values_by_ip(df, 'channel')
facets = ['n_os', 'n_app', 'n_channel', 'n_device', 'total_clicks', 'total_attributions']

combinations = [c for c in itertools.combinations(facets, 2)]
rows = 5
cols = int(len(combinations) / rows)

fig, axes = plt.subplots(rows, cols)
fig.set_figheight(20)
fig.set_figwidth(20)
fig.tight_layout()

idx = 0
for row in range(0, rows):
    for col in range(0, cols):
        combo = combinations[idx]
        ax = df.plot.hexbin(combo[0], combo[1], ax=axes[row, col], gridsize=22)
        idx+=1

