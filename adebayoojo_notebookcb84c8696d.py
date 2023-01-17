!pip install jovian opendatasets --upgrade --quiet
# Change this
dataset_url = 'https://www.kaggle.com/adebayoojo/global-earthquake-m6' 
import opendatasets as od
od.download(dataset_url)
# Change this
data_dir = './'
import os
os.listdir(data_dir)
project_name = "global-earthquakes" # change this (use lowercase letters and hyphens only)
!pip install jovian --upgrade -q
import jovian
jovian.commit(project=project_name)
# Here, I import neccessary libraries that we need in subsequent cells
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
# I use the pandas libray to import the earthquake data into a dataframe
df = pd.read_csv('./global-earthquake-m6/earthquake.csv', sep =  "|",header=0,index_col=1)
# I view the first few lines of the dataframe
df.head()
#I use the decribe property to get a feel of the original dataset
df.describe
df.shape
#Here I perform a bit of data cleaning. We do not need all the columns for our analysis, so we drop some columns nor needed
#We also check to see if the final dataframe has some missing data.
print(df.columns)
dff = df.filter([' Time ', ' Latitude ', ' Longitude ', ' Depth/km ',' Magnitude '], axis=1)
dff.index = pd.to_datetime(df.index)
dff.head()
dff.shape
dff.describe()
#From the information above, It is obvious that some depth column with 0 values are not accurate. 
#The minimum should be greater than zero for accurate dataset. So we need to fix this by removing this rows to create a final dataframe for analysis
bad = dff[dff[' Depth/km ']==0]
bad.shape
dff.drop(bad.index, inplace=True) #removing rows with zeo depths
dff.describe()
## We check for Null values using isnull() function
dff.isnull().sum()
import jovian
jovian.commit()
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
dff.describe()
sns.pairplot(dff)
mag_count = dff.groupby(' Magnitude ')[' Magnitude '].count().sort_values(ascending=False)
mag_count
dff.groupby(' Magnitude ')[' Magnitude '].count().sort_values(ascending=False).plot(kind='barh')
sns.countplot(x=' Magnitude ', data=dff)
depth_count = dff.groupby(' Depth/km ')[' Depth/km '].count().sort_values(ascending=False)
depth_count
dff.corr()
sns.heatmap(dff.corr(), annot = True)
import jovian
jovian.commit()
#Answering this question requires that we create a new columns with the event year for each row
df3 = dff.reset_index() #this will reset the index so that the time column is available 
df3["eqyear"] =  df3[' Time '].dt.year #this will get the year from the datetime object in the Time column
df3['eqyear'] = df3.eqyear.astype(int) #this will convert it from object to integer
print(df3.eqyear.dtypes) #this prints the data types
df3.head(3) # output the first three lines
eqyr_count = df3.groupby('eqyear')['eqyear'].count().sort_values(ascending=False) #regroup the dataframe using eqyear, count the frequency and sort
eqyr_count.idxmax() #get the year with maximum count
eqyr_count
(max(eqyr_count)/eqyr_count.sum())*100 #calculate the percentage the largest earthquake year (2011) represents
#Here I plot a map of the yearly distribution of earthquakes using seaborn
plt.figure(figsize=(30, 15))
plt.title('Number of earthquakes per year from 1999-2020',fontweight=800)
sns.countplot(x='eqyear', data=df3)
mgyr_count = df3.groupby('eqyear')[' Magnitude '].max().sort_values(ascending=False)
mgyr_count
#eqyr_count.idxmax()
mgyr_count.max()
sns.scatterplot(data=mgyr_count)
ax = plt.gca()
ax.set_title("Largest Magnitude every year")
#verifying the correlation between the frequency of events and maximum magnitude
b1 = df3.groupby('eqyear')[' Magnitude '].max()
b2 = df3.groupby('eqyear')['eqyear'].count()
b3 = pd.concat([b1,b2], axis=1)
b3.corr()


sns.heatmap(b3.corr(), annot = True)
df3[' Magnitude '].idxmax() # the idxmax() method returns the row number (location) of the maximum magnitude
df3.iloc[df3[' Magnitude '].idxmax()][' Time '] #I retrieve the time from the information
print('Largest Magnitude Earthquake Occurred on:\n','Date/Time:', df3.iloc[df3[' Magnitude '].idxmax()][' Time '],'\n', 'Magnitude:', df3.iloc[df3[' Magnitude '].idxmax()][' Magnitude '],'\n Latitude/Longitude:',df3.iloc[df3[' Magnitude '].idxmax()][' Latitude '],',',df3.iloc[df3[' Magnitude '].idxmax()][' Longitude '])
#Answering this question requires that we create a new columns with the event year for each row
df3["eqhour"] =  df3[' Time '].dt.hour #this will get the year from the datetime object in the Time column
df3['eqhour'] = df3.eqhour.astype(int) #this will convert it from object to integer
print(df3.eqhour.dtypes) #this prints the data types
df3.head(3) # output the first three lines
eqhr_count = df3.groupby('eqhour')['eqhour'].count() #regroup the dataframe using eqhour, count the frequency and sort
eqhr_count
#Here I plot a map of the yearly distribution of earthquakes using seaborn
plt.figure(figsize=(30, 15))
plt.title('Number of earthquakes in each hour of the day from 1999-2020',fontweight=800)
sns.countplot(x='eqhour', data=df3)
plt.figure(figsize=(20, 10))
df3.groupby('eqhour')['eqhour'].count().plot.pie(autopct="%.1f%%")

#The amount of energy is related to the earthquake magnitude. Generally larger earthquakes radiate more energy.
#I will compute the seismic moment from the magnitude using the formula of Hanks & Kanamori (1979)

#Answering this question requires that we create a new columns with the seismic moment for each row using a function
def bayo(x): 
    return 10**(1.5*(x+6.03))
df3["moment"] =  df3[' Magnitude '].apply(bayo) # I applied the function to the magnitude to get the moment
df3['moment'] = df3.moment.astype(float) #this will convert it from object to float
print(df3.moment.dtypes) #this prints the data types
df3 # output the first three lines
mon = df3.groupby('moment')['moment'].sum()
mon.sum()
yr_r=df3.groupby('eqyear')['moment'].sum()
yr_r
plt.figure(figsize=(20, 10))
df3.groupby('eqyear')['moment'].sum().plot.pie(autopct="%.1f%%")
import jovian
jovian.commit()
import jovian
jovian.commit()
import jovian
jovian.commit()
