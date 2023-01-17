# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/scrubbed.csv', low_memory=False)

df['datetime'] = pd.to_datetime(df['datetime'],errors='coerce')

df = df.dropna()
df['year'] = (df.datetime.dt.year).astype('int')

df = df.rename(columns={'longitude ':'longitude'})

df = df.rename(columns={'duration (seconds)':'duration'})

df['shape'] = df['shape'].astype('str')
dfus = df[df.country == 'us']

sns.distplot(dfus['year'], kde=False);
a = np.unique(dfus['shape'])

b = a[2]

print (b)

dfus_x = dfus[dfus['shape'] == b]

sns.distplot(dfus_x['year'], kde=False, );

#sns.swarmplot(x="year", hue='shape', data=dfus);