import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
sns.set_style('whitegrid')
df = pd.read_csv('../input/daily-inmates-in-custody.csv')
df.head()
df.info()
df.drop('DISCHARGED_DT', axis = 1, inplace = True)
df.dropna(axis = 0, inplace = True)
df.info()
df.head()
from datetime import datetime
df['ADMITTED_DT'] = df['ADMITTED_DT'].apply(lambda x: datetime.strptime(x,"%Y-%m-%dT%H:%M:%S"))
df['admitted_year'] = df['ADMITTED_DT'].apply(lambda x: x.year)
df['admitted_month'] = df['ADMITTED_DT'].apply(lambda x: x.month)
df.head()
sns.countplot(x = 'admitted_year', data = df)
sns.catplot(x = 'admitted_year', kind = 'count', data = df, hue = 'admitted_month', height = 5, aspect = 1.4)
current_year = datetime.now().year
# Get the age at the time when the inmates were admitted
df['admitted_age'] = df['AGE'] - (current_year - df['admitted_year'])
sns.catplot(x = 'admitted_year', y = 'admitted_age', kind = 'bar', data = df, height = 5, aspect = 1.4, ci = None)
df['admitted_year'].value_counts()
