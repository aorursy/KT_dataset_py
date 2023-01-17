import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("/kaggle/input/basketball-players-stats-per-season-49-leagues/players_stats_by_season_full_details.csv")
data.head()
data.info()
data.corr()
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.columns
# Line Plot
data.AST.plot(kind = 'line', color = 'g',label = 'AST',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.STL.plot(color = 'r',label = 'STL',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     
plt.xlabel('x axis')              
plt.ylabel('y axis')
plt.title('Line Plot')      
plt.show()
# Scatter Plot 
# x = FGM, y = FGA
data.plot(kind='scatter', x='FGM', y='FGA',alpha = 1,color = 'red')
plt.xlabel('FGM')            
plt.ylabel('FGA')
plt.title('FGM FGA Scatter Plot')            
# Histogram
data.AST.plot(kind = 'hist',bins = 50,figsize = (10,10))
plt.show()
data = pd.read_csv("/kaggle/input/basketball-players-stats-per-season-49-leagues/players_stats_by_season_full_details.csv")
data.head()
data.tail()
data.columns
data.shape
data.info()
# For example lets look frequency of player_stats League
print(data['League'].value_counts(dropna =False))  # if there are nan values that also be counted
# As it can be seen below there are 4136 Eurocup or 977 Turkish-BSL
data.describe().T
# For example: compare birth_year of player_stats that are nationality  or not
data.boxplot(column='birth_year',by = 'nationality')
# Firstly I create new data from player_stats data to explain melt nore easily.
data_new = data.head()    
data_new
# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame=data_new,id_vars = 'Player', value_vars= ['height_cm','weight_kg'])
melted
# Index is name
# I want to make that columns are variable
# Finally values in columns are value
melted.pivot(index = 'Player', columns = 'variable',values='value')
# Firstly lets create 2 data frame
data1 = data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row
data1 = data['height_cm'].head()
data2= data['weight_kg'].head()
conc_data_col = pd.concat([data1,data2],axis =1) # axis = 1 : adds dataframes in column
conc_data_col
data.dtypes
# lets convert object(str) to categorical and int to float.
data['Stage'] = data['Stage'].astype('category')
data['AST'] = data['AST'].astype('float')
data.dtypes
# Lets look at does pokemon data have nan value
# As you can see there are 53798 entries. However Team has 53787 non-null object so it has 11 null object.
data.info()
# Lets check high_school
data["high_school"].value_counts(dropna =False)
# As you can see, there are 30247 NAN value
# Lets drop nan values
data1=data   # also we will use data to fill missing value so I assign it to data1 variable
data1["high_school"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
# So does it work ?
#  Lets check with assert statement
# Assert statement:
assert 1==1 # return nothing because it is true
assert  data['high_school'].notnull().all() # returns nothing because we drop nan values
data["high_school"].fillna('empty',inplace = True)
assert  data['high_school'].notnull().all()