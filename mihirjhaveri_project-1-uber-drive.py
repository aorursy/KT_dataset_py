# Import the libraries and the dataset to datafram - uberdrive_df
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
# output is displayed in this juptyer notebook (inline dislay)
import matplotlib.pyplot as plt #collection of command style functions that make matplotlib work like MATLAB & data visulization
import seaborn as sns #Seaborn is a Python data visualization library based on matplotlib.

uberdrive_df = pd.read_csv('../input/uber-drive-project/uberdrive.csv') # Get the data
#View the last 10 rows of data
uberdrive_df.tail(10)
uberdrive_df.head(10) # View first 10 rows of data 
print(uberdrive_df.shape)
# size of data 
print (uberdrive_df.size)
# informatio of variables of the data set 
print (uberdrive_df.info())
#Is any of the values in the df null ?  ( # Useful in writing validation scripts on large number of files )
uberdrive_df.isnull().any().any()
# Same as above, gives non-null number of records
uberdrive_df.count()
# this gives the total count of the null(missing value) in the dataset
uberdrive_df.isnull().values.sum()
#Is any of the values in columns of the df null ? ( # Useful in writing validation scripts on large number of files )
uberdrive_df.isnull().any() 
uberdrive_df_null_cols = uberdrive_df.columns[uberdrive_df.isnull().any()]
uberdrive_df_null_cols = list(uberdrive_df_null_cols)
uberdrive_df_null_cols 
#Get the columns into a list and do use it to do some operations 
# this gives the total count of the null(missing value) in each column
uberdrive_df.isnull().sum()  #Shows  the column wise values of missing data
# Get the initial data with dropping the NA values, used , the name uberdrive_df_dropped instead of df, more descriptive 
uberdrive_df_dropped = uberdrive_df.dropna()  

#Is any of the values in columns of the df null ? 
print(uberdrive_df_dropped .isnull().values.any())

#Get the shape of the dataframe after removing the null values
print (uberdrive_df_dropped.shape)
#to get the count (via shape) details before drop 
print(uberdrive_df.shape)
# to get the summary of the original data
uberdrive_df.describe()
# informatio of variables of the new dropped data set 
uberdrive_df_dropped.info()
uber_driver_start_destination = uberdrive_df["START*"].dropna()
unique_uber_driverstart = set(uber_driver_start_destination ) # set function provides unique values
print(unique_uber_driverstart)
# I also used an alternate approach, the result is different, as worked only on the second dataset, where all the na dropped

# Get the unique starting point, unique destination
# names of unique start points
print(uberdrive_df_dropped['START*'].unique())
len(unique_uber_driverstart) # use the len function to get the total number of unique from above
# or use can use the nunique function #this also gives the count of unique values in START column
print(uberdrive_df['START*'].nunique())

# again this is additional, did on the new dataset , with all dropped NA value, so the length is different, 
#this is for my understanding only.
len(uberdrive_df_dropped['START*'].unique()) #count of unique start points using  len()
uber_driver_stop_destination = uberdrive_df["STOP*"].dropna()
unique_uber__stop = set(uber_driver_stop_destination)
print(unique_uber__stop)
len(unique_uber__stop)
# again this is additional, did on the new dataset , with all dropped NA value, so the length is different, 
#this is for my understanding only.
print(len(uberdrive_df_dropped['STOP*'].unique())) #count of unique stop points
uberdrive_df_Start_San_Francisco = uberdrive_df[uberdrive_df['START*']== 'San Francisco']
uberdrive_df_Start_San_Francisco

#alternate way for the above questions.
uberdrive_df.loc[uberdrive_df["START*"] == "San Francisco"]
#Identify popular start points - top 10
uberdrive_df['START*'].value_counts().head(10)
uberdrive_df['START*'].value_counts().head(1)
#alternate way, different approach

uber_driver_starting_point = uberdrive_df["START*"].dropna()
uber_driver_starting_point_dropped_na = pd.DataFrame(uber_driver_starting_point.value_counts())
uber_driver_starting_point_dropped_na.sort_values(["START*"], ascending = False)

uber_driver_starting_point_dropped_na = uber_driver_starting_point_dropped_na.reset_index()
uber_driver_starting_point_dropped_na = uber_driver_starting_point_dropped_na.rename(columns = {'index':'starting_destination', 'START*':'Count'})
uber_driver_starting_point_dropped_na.loc[uber_driver_starting_point_dropped_na['Count'] == max(uber_driver_starting_point_dropped_na['Count'])]
#Identify popular stop destinations - top 10
uberdrive_df['STOP*'].value_counts().head(10)
uberdrive_df['STOP*'].value_counts().head(1)
#alternate way, different approach for the above
uber_driver_stopping_point = uberdrive_df["STOP*"].dropna()
uber_driver_stopping_point_dropped_na = pd.DataFrame(uber_driver_stopping_point.value_counts())
uber_driver_stopping_point_dropped_na.sort_values(["STOP*"], ascending = False)

uber_driver_stopping_point_dropped_na = uber_driver_stopping_point_dropped_na.reset_index()
uber_driver_stopping_point_dropped_na = uber_driver_stopping_point_dropped_na.rename(columns = {'index':'stop_destination', 'STOP*':'Count'})
uber_driver_stopping_point_dropped_na.loc[uber_driver_stopping_point_dropped_na['Count'] == max(uber_driver_stopping_point_dropped_na['Count'])]


uberdrive_df.groupby(['START*','STOP*'])['MILES*'].sum().sort_values(ascending=False).head()
#Dropping Unknown Location Value  - Save into anothe dataframe ( you dont want to overwrite the original df)
uberdrive_df_drop_unknown_location = uberdrive_df[uberdrive_df['START*']!= 'Unknown Location']
uberdrive_df_drop_unknown_location = uberdrive_df_drop_unknown_location[uberdrive_df_drop_unknown_location['STOP*']!= 'Unknown Location']

uberdrive_df_drop_unknown_location.groupby(['START*','STOP*'])['MILES*'].sum().sort_values(ascending=False).head(10)
#second approach 
#The most popular start and stop pair - ( BY COUNT of travels! )
uberdrive_df_drop_unknown_location.groupby(['START*','STOP*'])['MILES*'].size().sort_values(ascending=False).head(10)
#third approach

df3 = uberdrive_df.dropna()
df3 = pd.DataFrame(df3.groupby(['START*', 'STOP*']).size())
df3 = df3.rename(columns = {0:'Count'})
df3 = df3.sort_values(['Count'], ascending = False)
df3.loc[df3['Count'] == max(df3['Count'])]
#using MILES
print(np.array(uberdrive_df['PURPOSE*'].dropna().unique()))
uberdrive_df['MILES*'].groupby(uberdrive_df['PURPOSE*']).sum()


#using count
uberdrive_df['PURPOSE*'].value_counts()
#Doing a quick plot 
k3 = uberdrive_df.groupby('PURPOSE*')['MILES*'].sum().sort_values(ascending=False).head(10) 
k3
k3= k3.reset_index() # flatten the dataframe 
k3
k3.columns = ['PURPOSE*' ,'sum_of_miles']
k3
%matplotlib inline 
import seaborn as sns
sns.barplot(data= k3 , x= 'PURPOSE*' , y ='sum_of_miles')
plt.xticks(rotation=90)
#another way
visual_df = pd.DataFrame(uberdrive_df['MILES*'].groupby(uberdrive_df['PURPOSE*']).sum())
visual_df .plot(kind = 'bar')
plt.show()
#another way
visual_df =  visual_df.reset_index()
sns.barplot(x = visual_df['MILES*'], y = visual_df['PURPOSE*'])
 #How many miles was earned per  purpose ?
uberdrive_df.groupby('PURPOSE*').sum()['MILES*'].sort_values(ascending = False)
#conituation from the above question 19, can use the same
visual_df
uberdrive_df['CATEGORY*'].value_counts()
sns.countplot(uberdrive_df['CATEGORY*'])
#another way
uberdrive_df.head()

visual_df2 = pd.DataFrame(uberdrive_df['CATEGORY*'].value_counts())
visual_df2.reset_index()

visual_df2.plot(kind = 'bar')
plt.show()
visual_df2
#How many miles was earned per category and purpose ?
uberdrive_df.groupby('CATEGORY*').sum()['MILES*'].sort_values(ascending = False)
#What is percentage of business miles vs personal?
uberdrive_df_cat_df = uberdrive_df.groupby('CATEGORY*').agg({'MILES*':'sum'})
uberdrive_df_cat_df
uberdrive_df_cat_df.apply(lambda x: x/x.sum()*100).rename(columns = {'MILES':'% of Miles'})