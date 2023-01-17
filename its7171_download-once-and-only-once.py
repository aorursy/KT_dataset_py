import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
nrows=None
dtypes = {
        'ip'            : 'uint32',
        'is_attributed' : 'uint8',
        }
train_df = pd.read_csv('../input/train.csv',dtype=dtypes,nrows=nrows,usecols=['ip','is_attributed'])
ip_grp = train_df[['ip','is_attributed']].groupby(['ip']).agg(['count', 'sum']).is_attributed.sort_values(by='count').reset_index()
ip_grp.columns = ['ip', 'occurrences', 'download_count']
df = ip_grp[['download_count','occurrences']].groupby('occurrences').agg(['count', 'sum'])['download_count']
df.columns = ['num_IPs', 'num_downloads']
df['cvr_x_occurrences'] = df.num_downloads /df.num_IPs
df = df.reset_index()
df.head(10)
thre = 400
_df = df[df.occurrences<thre].copy()
plt.plot(_df.occurrences, _df.cvr_x_occurrences)
train_df[train_df.ip.isin(ip_grp[ip_grp.occurrences==1].head(10).ip)].sort_values(['ip','is_attributed'])
train_df[train_df.ip.isin(ip_grp[ip_grp.occurrences==3].head(10).ip)].sort_values(['ip','is_attributed'])
thre = 3000
_df = df[df.occurrences<thre].copy()
_df['roll'] = _df.cvr_x_occurrences.rolling(window=int(10)).mean()
plt.plot(_df.occurrences, _df.roll)
sum_download = sum(df.num_downloads)
df['cum_download'] = df.num_downloads.cumsum()
df['cum_download_ratio'] = df['cum_download']/sum_download
df['pv'] = df.num_IPs*df.occurrences
sum_pv = sum(df.pv)
df['cum_pv'] = df.pv.cumsum()
df['cum_pv_ratio'] = df.cum_pv / sum_pv
df['cvr'] = df.num_downloads / df.pv
df[['occurrences','cum_download_ratio','cum_pv_ratio']].head(50)
