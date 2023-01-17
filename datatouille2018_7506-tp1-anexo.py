import pandas as pd
import numpy as np

df = pd.read_csv("../input/events.csv", low_memory=False)
df['timestamp'] = pd.to_datetime(df['timestamp'])
checked = ['iphone', 'samsung', 'motorola', 'lenovo', 'sony', 'lg', 'ipad', 'asus', 'quantum', 'blackberry']
model_parsed = df['model'].dropna().map(lambda x: x.lower())
model_parsed = model_parsed.map(lambda x: x.split())

def find_brand(model):
    for str in model:
        if str in checked:
            return str
    return "other"


df['brand'] = model_parsed.map(find_brand)
df['brand'] = df['brand'].astype('category')
#df[['model', 'brand']].dropna().drop_duplicates().to_csv('data/brands.csv', index=False)
brands_csv = pd.read_csv("../input/brands.csv", low_memory=False)
display(brands_csv.head())
brands_csv.shape
df['operating_system_version'].unique()
checked = ['windows', 'android', 'linux', 'mac', 'ios', 'ubuntu', 'chrome os', 'tizen', 'other']
os_version_parsed = df['operating_system_version'].dropna().map(lambda x: x.lower())

def find_os(os_version):
    for os in checked:
        if os in os_version:
            return os
    return "another"


df['operating_system'] = os_version_parsed.map(find_os)
df['operating_system'] = df['operating_system'].astype('category')
# Chequeamos cuantos os quedaron con el nombre 'another' (idealmente, ninguno)
df[['operating_system_version', 'operating_system']].dropna().head(10)
df[df['operating_system'] == 'another'][['operating_system_version', 'operating_system']].head(10)
df['operating_system'].value_counts()
#df[['operating_system_version', 'operating_system']].dropna().drop_duplicates().to_csv('data/os.csv', index=False)
os_csv = pd.read_csv("../input/os.csv", low_memory=False)
display(os_csv.head())
os_csv.shape
checked = ['mobile safari', 'chrome mobile', 'ie mobile', 'firefox mobile', 'edge mobile', 'opera mobile',
           'mobile', 'chrome', 'android', 'opera', 'ie', 'firefox', 
           'facebook', 'samsung', 'chromium', 'edge', 'yandex', 'uc', 
           'other', 'safari', 'puffin', 'maxthon', 'vivaldi']
browser_version_parsed = df['browser_version'].dropna().map(lambda x: x.lower())

def find_browser(browser_version):
    for browser in checked:
        if browser in browser_version:
            return browser
    return "other"


df['browser'] = browser_version_parsed.map(find_browser)
df['browser'] = df['browser'].astype('category')
df[['browser_version', 'browser']].dropna()
display(df['browser'].value_counts().head())
df[['browser_version', 'browser']].head()
display(df[['browser_version', 'browser']].dropna().drop_duplicates().head())
#df[['browser_version', 'browser']].dropna().drop_duplicates().to_csv('data/browsers.csv', index=False)
browsers_csv = pd.read_csv("../input/browsers.csv", low_memory=False)
display(browsers_csv.head())
browsers_csv.shape
funnel = df.sort_values(['person', 'timestamp'])
funnel.head()
funnel['time_diff'] = funnel.groupby('person')['timestamp'].diff()
funnel['time_diff'] = funnel['time_diff'].fillna(0)
funnel['time_diff_min'] = funnel['time_diff'] / np.timedelta64(1, 'm')
THRESHOLD = 30 # minutes

funnel['new_session'] = funnel['time_diff_min'] > THRESHOLD
funnel['session_id'] = funnel.groupby('person')['new_session'].cumsum()
funnel['session_id'] = funnel['session_id'].astype('int')
gb = funnel.groupby(['person', 'session_id'])

funnel['session_cumno'] = gb.cumcount()
funnel['session_total_events'] = gb['session_cumno'].transform(lambda x: x.size)
funnel['session_first'] = funnel['session_cumno'] == 0
funnel['session_last'] = funnel['session_cumno'] == (-1+funnel['session_total_events'])
cols = ['person', 'timestamp', 'time_diff_min', \
        'session_id', 'event', 'session_total_events', \
        'session_cumno', 'session_first', 'session_last']
funnel[cols]
funnel['is_conversion'] = funnel['event'] == 'conversion'
gb = funnel.groupby(['person', 'session_id'])
funnel['session_conversion'] = gb['is_conversion'].transform(lambda x: x.sum())
funnel['session_conversion'] = funnel['session_conversion'] > 0
funnel.head()
funnel['is_checkout'] = funnel['event'] == 'checkout'
gb = funnel.groupby(['person', 'session_id'])
funnel['session_checkout'] = gb['is_checkout'].transform(lambda x: x.sum())
funnel['session_checkout'] = funnel['session_checkout'] > 0
funnel.head()
funnel['ad_origin'] = (funnel['event'] == 'ad campaign hit') & funnel['session_first']
gb = funnel.groupby(['person', 'session_id'])
funnel['session_ad'] = gb['ad_origin'].transform(lambda x: x.sum())
funnel['session_ad'] = funnel['session_ad'] > 0
funnel.head()
funnel[cols]
cols_csv = ['time_diff_min', \
        'session_id', 'session_total_events', \
        'session_cumno', 'session_first', 'session_last', \
        'session_conversion', 'session_checkout', 'session_ad']

#funnel[cols_csv].to_csv('data/sessions.csv', index=False)
sessions_csv = pd.read_csv("../input/sessions.csv", low_memory=False)
display(sessions_csv.head())
sessions_csv.shape