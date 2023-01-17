import pandas as pd
loc = '../input/cpe-data/Dept_37-00027/37-00027_UOF-P_2014-2016_prepped.csv'
df = pd.read_csv(loc, header=[1])
df['Date Occurred'] = pd.to_datetime(df['Date Occurred'])
df.head().T
df.shape
pct_nan = df.isnull().sum() / df.shape[0]
pct_nan = pct_nan[pct_nan > 0.01]
pct_nan.name = "nan"
pct_nan.to_frame().style \
    .format("{:.1%}")
df['RIN'].nunique()
df['Primary Key'].nunique()
df[df['Primary Key'] == 2015541517]

# same incident, different people (:
df['Date Occurred'].min()
df['Date Occurred'].max()
df['Date Occurred'].dt.year.value_counts()
df['Area Command'].value_counts()
df['Nature of Contact'].value_counts()
df['Reason Desc'].value_counts()
df['Subject Sex'].value_counts()
df['Race'].value_counts()
df['Subject Role'].value_counts().head(10)
df['Subject Conduct Desc'].value_counts()
df['Subject Resistance'].value_counts().head(10)
# needs cleaning (?)
df['Weapon Used 1'].value_counts()
df['Weapon Used 2'].value_counts()
df['Weapon Used 3'].value_counts()
df['Weapon Used 4'].value_counts()
df['Number Shots'].value_counts()
df['Subject Effects'].value_counts()
df['Effect on Officer'].value_counts()
df['Officer Organization Desc'].value_counts().head(10)
df['Officer Yrs of Service'].value_counts().head()
df['X-Coordinate'].nunique()
df['Y-Coordinate'].nunique()
df['City Council District'].value_counts()
df['Geolocation'].nunique()
df['City'].value_counts()
df['State'].value_counts()
df['Latitude'].nunique()
df['Longitude'].nunique()