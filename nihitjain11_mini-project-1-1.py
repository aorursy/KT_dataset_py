# get the libraries for analysis and model building
import pandas as pd
import numpy as np
import statsmodels.api as sm;
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

# load the dataset as DataFrame
df = pd.read_csv('../input/Crash.dat',sep='\t') 

# View the first 5 rows of the dataset
df.head()
# dataset size , i.e., number of rows x columns
df.shape
# display the name of the columns in the dataset
df.columns
# Total number of unique values under each column
df.nunique()
# short summary of the dataset, i.e.,
# col_name    data_count    value_type    data_type
df.info()
df.describe()
# Total count of duplicate rows in the dataset
df.duplicated().sum()
# Count of null values in the dataset
df.isnull().sum()
df.dtypes

# Fill in empty spaces with NaN values by extracting values from strings
# Columns = HEAD_INJ, CHEST_INJ,LLEG_INJ, RLEG_INJ, DOORS, PROTECT2
df['HEAD_INJ'] = df['HEAD_INJ'].str.extract('(\d+)')
df['CHEST_IN'] = df['CHEST_IN'].str.extract('(\d+)')
df['LLEG_INJ'] = df['LLEG_INJ'].str.extract('(\d+)')
df['RLEG_INJ'] = df['RLEG_INJ'].str.extract('(\d+)')
df['DOORS']    = df['DOORS'].str.extract('(\d+)')
df['PROTECT2'] = df['PROTECT2'].str.extract('(\d+)')

df.head()
# Count of null values in the dataset (After conversion from empty string to NaN)
df.isnull().sum()
# Dropping all rows with null values in HEAD_INJ column
df.dropna(subset=['HEAD_INJ'],inplace=True)
# Convert object(string) type columns into integer type
df['HEAD_INJ'] = df['HEAD_INJ'].astype('int64')
# fill NaN in other columns with average value of respective columns
# 'CHEST_IN','LLEG_INJ','RLEG_INJ','DOORS','PROTECT2'

val = df['CHEST_IN'].dropna() # drop NaN value in the column
val = pd.to_numeric(val)       # convert the col values to numeric
mean_val = round(val.mean())       # get rounded mean of the column
df['CHEST_IN'].fillna(mean_val,inplace=True) # fillin the Mean Value in NaN of actual column

# applying smae steps for other columns

val = df['LLEG_INJ'].dropna() 
val = pd.to_numeric(val)      
mean_val = round(val.mean())  
df['LLEG_INJ'].fillna(mean_val,inplace=True)

val = df['RLEG_INJ'].dropna() 
val = pd.to_numeric(val)    
mean_val = round(val.mean())       
df['RLEG_INJ'].fillna(mean_val,inplace=True) 

val = df['DOORS'].dropna() 
val = pd.to_numeric(val)       
mean_val = round(val.mean())   
df['DOORS'].fillna(mean_val,inplace=True) 

val = df['PROTECT2'].dropna() 
val = pd.to_numeric(val)      
mean_val = round(val.mean())  
df['PROTECT2'].fillna(mean_val,inplace=True) 

df[['CHEST_IN','LLEG_INJ','RLEG_INJ','DOORS','PROTECT2']].isnull().sum()
# Convert object(string) type columns into integer type
df[['HEAD_INJ','CHEST_IN','LLEG_INJ','RLEG_INJ','DOORS','PROTECT2']] = df[['HEAD_INJ','CHEST_IN','LLEG_INJ',
                                                                           'RLEG_INJ','DOORS','PROTECT2']].astype('int64')
df.dtypes
df.describe()
df['intercept'] = 1
lm = sm.OLS(df['HEAD_INJ'],df[['intercept','CHEST_IN','LLEG_INJ','RLEG_INJ','DOORS','YEAR','WEIGHT','SIZE2','PROTECT2']])
results = lm.fit()
results.summary()
sns.pairplot(df[['HEAD_INJ','CHEST_IN','LLEG_INJ','RLEG_INJ','DOORS','PROTECT2']]);
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
y,X = dmatrices('HEAD_INJ~intercept+CHEST_IN+LLEG_INJ+RLEG_INJ+DOORS+YEAR+WEIGHT+SIZE2+PROTECT2',df,return_type='dataframe')
vif = pd.DataFrame()
vif['vifFactor'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['features'] = X.columns
vif
pd.plotting.scatter_matrix(df,figsize=(20,20));
df.plot(kind='scatter',x='HEAD_INJ',y='YEAR',figsize=(15,15),grid=True)
# Which company make had highest head injury count
a = df.groupby(['MAKE'],as_index=False).sum()
m = (a["HEAD_INJ"].max())
a[a['HEAD_INJ']>=m]
# Which company make had lowest head injury count
a = df.groupby(['MAKE'],as_index=False).sum()
m = (a["HEAD_INJ"].min())
a[a['HEAD_INJ']<=m]
# Which year car make had the highest head injury
a = df.groupby(['YEAR'],as_index=False).sum()
m = (a["HEAD_INJ"].max())
a[a['HEAD_INJ']>=m]
# WHch year car make had the lowest head injury
a = df.groupby(['YEAR'],as_index=False).sum()
m = (a["HEAD_INJ"].min())
a[a['HEAD_INJ']<=m]
# whihc TYpes of protection have highest influence on head injury
a = df.groupby(['PROTECT2'],as_index=False).sum()
m = (a["HEAD_INJ"].max())
a[a['HEAD_INJ']>=m]

# whihc TYpes of protection have least influence on head injury
a = df.groupby(['PROTECT2'],as_index=False).sum()
m = (a["HEAD_INJ"].min())
a[a['HEAD_INJ']<=m]

# TYpes of protection have influence on which type of injury

# Significance of doors on  head injury
a = df.groupby(['DOORS'],as_index=False).sum()
m = (a["HEAD_INJ"].max())
a[a['HEAD_INJ']>=m]
plt.bar(a['DOORS'],a['HEAD_INJ']);
# Significance of door on  various injury

# Driver had more injury or passenger (line graph plot of driver and passenger based on various factors)
a = df.groupby(['DRIV_PAS'],as_index=False).mean()
m = (a["HEAD_INJ"].max())
a
df[['HEAD_INJ','CHEST_IN','LLEG_INJ','RLEG_INJ']].describe()
plt.figure(figsize=(12,8))
sns.lineplot(df['YEAR'],df['HEAD_INJ'],df['DRIV_PAS'],palette="tab10", linewidth=2.5).set_title('Head Injury Over the Years');
plt.figure(figsize=(12,8))
sns.lineplot(df['YEAR'],df['CHEST_IN'],df['DRIV_PAS'],palette="tab10", linewidth=2.5).set_title('Chest Injury Over the Years');
plt.figure(figsize=(12,8))
sns.lineplot(df['YEAR'],df['LLEG_INJ'],df['DRIV_PAS'],palette="tab10", linewidth=2.5).set_title('Left Leg Injury Over the Years');
plt.figure(figsize=(12,8))
sns.lineplot(df['YEAR'],df['RLEG_INJ'],df['DRIV_PAS'],palette="tab10", linewidth=2.5).set_title('Right Leg Injury Over the Years');
# Over the years have the car make company have improved in safety measures or not. 

# Over the years have the car make company have improved in safety  of head injury measures or not. 


# Over the years have the injury safety improved ?

# Over the years have the head injury safety improved ? And which injury has declined?
plt.figure(figsize=(12,8))
sns.lineplot(df[['HEAD_INJ','CHEST_IN','LLEG_INJ','RLEG_INJ']],df['YEAR'],columns=['HEAD_INJ','CHEST_IN','LLEG_INJ','RLEG_INJ'],palette="tab10", linewidth=2.5).set_title('Injuries Over the Years');

rs = np.random.RandomState(365)
values = rs.randn(365, 4).cumsum(axis=0)
dates = pd.date_range("1 1 2016", periods=365, freq="D")
data = pd.DataFrame(values, dates, columns=["A", "B", "C", "D"])
data = data.rolling(7).mean()
values
# Does car size affect head injury or other inury?

# no. of cars model of a single company and thier saftey measure as improved over the year


# Does other injury help prevent other injury


# which company make is the safest


# which company model is the safest


# which year is the safest

