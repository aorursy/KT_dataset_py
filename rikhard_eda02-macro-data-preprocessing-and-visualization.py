import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
# Load train data

df = pd.read_csv('../input/macro.csv')
print(df.columns.tolist())

print('\nNumber of columns on train data:',len(df.columns))

print('\nNumber of data points:',len(df))

print('\nNumber of unique timestamp data points:',len(df['timestamp'].unique()))

print('\nData types:',df.dtypes.unique())
df.select_dtypes(include=['O']).columns.tolist()
df.select_dtypes(include=['O']).dropna().head()
object_cols = df.select_dtypes(include=['O']).columns.tolist()[1:]

for col in object_cols:

    df[col] = df[col].apply(lambda s: str(s).replace(',','.'))

df[object_cols] = df[object_cols].apply(pd.to_numeric, errors='coerce')

df[object_cols].dropna().head()
df.select_dtypes(include=['O']).columns.tolist()
# Get the number of NaN's for each column, discarding those with zero NaN's

ranking = df.loc[:,df.isnull().any()].isnull().sum().sort_values()

# Turn into %

x = ranking.values/len(df)



# Plot bar chart

index = np.arange(len(ranking))

plt.bar(index, x)

plt.xlabel('Features')

plt.ylabel('% NaN observations')

plt.title('% of null data points for each feature')

plt.show()



print('Features:',ranking.index.tolist())

print('\nNumber of columns which have any NaN:',df.isnull().any().sum(),'/',len(df.columns))

print('\nNumber of rows which have any NaN:',df.isnull().any(axis=1).sum(),'/',len(df))
df.iloc[1000:1010,:10]
df['year'] = df['timestamp'].apply(lambda f: f.split('-')[0])

df['month'] = df['timestamp'].apply(lambda f: f.split('-')[1])

df['day'] = df['timestamp'].apply(lambda f: f.split('-')[2])

df['quarter'] = np.floor((df['month'].values.astype('int')-1)/3).astype('int')+1

df['year-quarter'] = df['year'] +'-Q' + df['quarter'].astype('str')

df['year-month'] = df['year'] +'-'+ df['month']

del df['quarter']
def get_periods(df):

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    var = df.columns[1]

    df['shift'] = df[var].shift()

    df = df.dropna()

    index = df[df[var]-df['shift']!=0].index

    if len(df.loc[:,var].unique())>3:

        return df.loc[index,'timestamp'].diff().mean().round('D')

    else: return -1
periods = list()

for i,col in enumerate(df.columns[1:-5]):

    periods.append(get_periods(df[['timestamp',col]].copy()))

    print(i,col,periods[i])
set(periods)
final_periods = dict()

for col, period in zip(df.columns[1:-5], periods):

    if period == -1: continue

    elif period <= pd.Timedelta('2 days'): final_periods[col] = 1

    elif period <= pd.Timedelta('35 days'): final_periods[col] = 30

    elif period <= pd.Timedelta('100 days'): final_periods[col] = 90

    elif period <= pd.Timedelta('370 days'): final_periods[col] = 365

    

print('Number of columns of interest:',len(final_periods))
%matplotlib inline

for key, value in final_periods.items():

    resume = pd.DataFrame(columns=[key,'mean'])

    if value == 1:

        resume[key] = df[key]

        resume['mean'] = resume[key].dropna().rolling(30,center=True).mean()

    elif value == 30:

        resume[key] = df.groupby('year-month').agg('mean')[key]

        resume['mean'] = resume[key].dropna().rolling(12,center=True).mean()

    elif value == 90:

        resume[key] = df.groupby('year-quarter').agg('mean')[key]

        resume['mean'] = resume[key].dropna().rolling(4,center=True).mean()

    elif value == 365:

        resume[key] = df.groupby('year').agg('mean')[key]

    resume.plot()

    