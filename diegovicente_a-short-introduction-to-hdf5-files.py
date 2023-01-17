import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
%matplotlib inline
with pd.HDFStore('../input/madrid.h5') as data:
    df = data['master']
    
df.head()
with pd.HDFStore('../input/madrid.h5') as data:
    for k in data.keys():
        print('{}: {}'.format(k, ', '.join(data[k].columns)))
with pd.HDFStore('../input/madrid.h5') as data:
    test = data['28079016']

test.rolling(window=24).mean().plot(figsize=(20, 7), alpha=0.8)
partials = list()

with pd.HDFStore('../input/madrid.h5') as data:
    stations = [k[1:] for k in data.keys() if k != '/master']
    for station in stations:
        df = data[station]
        df['station'] = station
        partials.append(df)
            
df = pd.concat(partials, sort=False).sort_index()

df.info()