import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import itertools
from subprocess import check_output

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('../input/attacks_data.csv',
                 encoding='latin1', parse_dates=['Date'],
                 infer_datetime_format=True,
                 index_col=1,
                )
df.info()
plt.figure(figsize=(15, 5))

plt.subplot(121)
df.Killed.plot()
plt.title('Killed')

plt.subplot(122)
df.Injured.plot()
plt.title('Injured')
df.Country.value_counts().plot(kind='bar', figsize=(17, 7))
plt.title('Number of attacks by countries')
(df['2015'].Country.value_counts() - 
df['2014'].Country.value_counts()).plot(kind='bar', figsize=(17, 7))
upto_month = str(df['2016':].index.month.max())
(
df['2016-' + upto_month].Country.value_counts() - 
df['2015-' + upto_month].Country.value_counts()
).plot(kind='bar', figsize=(17, 7))
df.groupby('Country').sum()[['Killed', 'Injured']].plot(kind='bar', figsize=(17, 7), subplots=True)
plt.figure(figsize=(15, 10))
years = list(set(df.index.year))
years.sort()
for index, year in enumerate(years):
    plt.subplot(4, 4, index+1)
    plt.hist(df[str(year)].index.month)
    plt.title(str(year))