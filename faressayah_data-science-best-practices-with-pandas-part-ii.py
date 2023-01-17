import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")
df = pd.read_csv("/kaggle/input/police_project.csv")
df.head()
df.info()
df.describe()
df.shape
df.isnull().sum()
df.dropna(axis=1, how='all').shape
df.drop('county_name', axis=1, inplace=True)
df.isnull().sum()
sns.catplot('driver_gender', data=df, kind="count", height=7)
df.driver_gender.value_counts()
print(df[df.violation == 'Speeding'].driver_gender.value_counts(normalize=True))
plt.figure(figsize=(12, 8))
df[df.violation == 'Speeding'].driver_gender.value_counts().plot(kind="bar")
df.loc[df.violation == "Speeding", "driver_gender"].value_counts(normalize=True)
df[df.driver_gender == "M"].violation.value_counts(normalize=True)
df[df.driver_gender == "F"].violation.value_counts(normalize=True)
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
df[df.driver_gender == "F"].violation.value_counts(normalize=True).plot(kind="bar")
plt.title("Violation of Women")

plt.subplot(2, 2, 2)
df[df.driver_gender == "M"].violation.value_counts(normalize=True).plot(kind="bar")
plt.title("Violation of Men")
sns.catplot('violation', data=df, hue='driver_gender', kind='count', height=8)
df.search_conducted.value_counts()
df.loc[df.search_conducted, 'driver_gender'].value_counts()
df.groupby(['violation', 'driver_gender']).search_conducted.mean()
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
df.search_conducted.value_counts().plot(kind="bar")
plt.title("Searched Cases")

plt.subplot(2, 2, 2)
df.loc[df.search_conducted, 'driver_gender'].value_counts().plot(kind="bar")
plt.title("Searched Men and Women")

plt.subplot(2, 2, 3)
df.groupby(['violation', 'driver_gender']).search_conducted.mean().plot(kind="bar")
df.search_type.isnull().sum()
df.search_conducted.value_counts()
df[df.search_conducted == False].search_type.value_counts(dropna=False)
df.search_type.value_counts()
plt.figure(figsize=(12, 8))
df.search_type.value_counts().plot(kind="bar")
df.search_type.value_counts()
counter = 0
for item in df.search_type:
    if type(item) == str and "Protective Frisk" in item:
        counter += 1
print(counter)
df.search_type.str.contains('Protective Frisk').sum()
df.search_type.str.contains('Protective Frisk').mean()
df.head()
print(df.stop_date.dtype)
print(df.stop_time.dtype)
df.stop_date
df['stop_date'] = pd.to_datetime(df.stop_date, format="%Y-%M-%d")
df["year"] = df.stop_date.dt.year
df.dtypes
df.year.value_counts()
plt.figure(figsize=(12, 8))
df.year.value_counts().plot(kind="bar")
df.columns
df.drugs_related_stop.value_counts()
df["stop_time"] = pd.to_datetime(df.stop_time, format="%H:%M").dt.hour
df.head()
df.loc[df.sort_values(by="stop_time").drugs_related_stop, 'stop_time'].value_counts()
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
df.loc[df.sort_values(by="stop_time").drugs_related_stop, 'stop_time'].value_counts().sort_index().plot(kind="bar")

plt.subplot(2, 2, 2)
df.loc[df.sort_values(by="stop_time").drugs_related_stop, 'stop_time'].value_counts().sort_index().plot()
df.stop_time.sort_index().value_counts().sort_index()
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
df.stop_time.sort_index().value_counts().sort_index().plot()

plt.subplot(2, 2, 2)
df.stop_time.sort_index().value_counts().sort_index().plot(kind="bar")
df.stop_duration.isnull().sum()
df.stop_duration.unique()
df.stop_duration.value_counts(dropna=False)
# ri.stop_duration.replace(['1', '2'], value=np.nan, inplace=True)
df.loc[(df.stop_duration == '1')| (df.stop_duration == '2'), 'stop_duration'] = np.nan
df.stop_duration.value_counts(dropna=False)
df.stop_duration.unique()
df.violation_raw.value_counts()
df.groupby('stop_duration').violation_raw.value_counts()
sns.catplot("stop_duration", data=df, hue="violation_raw", kind="count", height=7)
plt.figure(figsize=(12, 12))
df.groupby('stop_duration').violation_raw.value_counts().plot(kind="bar")
mapping = {'0-15 Min':8, '16-30 Min':23, '30+ Min':45}
df['stop_minutes'] = df.stop_duration.map(mapping)
df.stop_minutes.value_counts()
df.groupby('violation_raw').stop_minutes.mean()
df.groupby('violation_raw').stop_minutes.agg(['mean', 'count'])
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
df.groupby('violation_raw').stop_minutes.mean().plot(rot=45)

plt.subplot(2, 2, 2)
df.groupby('violation_raw').stop_minutes.mean().plot(kind="bar")
df.groupby("violation").driver_age.describe()
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
df.driver_age.hist(bins=10)

plt.subplot(2, 2, 2)
df.driver_age.value_counts().sort_index().plot()
df.hist('driver_age', by='violation', figsize=(12, 12));