# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import Imputer #tool to replace missing value
import matplotlib.pyplot as plt #plotting charts
import seaborn as sns #plotting good-looking charts
import cufflinks as cf #plotting interative charts
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot 
init_notebook_mode(connected=True) #connect the javescript to the notebook
cf.go_offline() #allow using cufflinks offline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#load data
df = pd.read_csv('../input/googleplaystore.csv')
#display the first 10 records
df.head(10)
#drop off irrelevant columns
df = df.drop(columns=['Last Updated', 'Android Ver','Content Rating'])
#check out the type and counts for each attribute
df.info()
#check out the nan value for each attribute
df.isna().sum()
#print the list of column names
df.columns.values
#find out the column with uncertain value "Varies with device"
for i in range(4,len(df.columns)):
    print (df.columns.values[i] + " " + str(df[df[df.columns.values[i]] == 'Varies with device'][df.columns.values[i]].count()))
    #print(df[df[df.columns.values[i]] == 'Varies with device'][df.columns.values[i]].count())
#df[df['Size'] == 'Varies with device'].Size.count()
def CleanName(str):
    "This function will remove some special symbols from a string"
    str = str.replace('- ','')
    str = str.replace('– ','')
    str = str.replace('& ','')
    str = str.replace(', ','')
    return str
#create a new column with app names without special symbols
df['Clean_Name'] = df.App.apply(lambda x:CleanName(x))
#double check if the function works
df.iloc[:5,10:11]
#create a attribute recording the number of words in an app name excluding special symbols
df['Name_Word_Counts'] = df.Clean_Name.str.count(' ')+1
#double check the attribute 'Name_Word_Counts'
df.iloc[:5,10:]
#create a attribute recording the number of characters in an app's name including special symbols
df['Name_Length'] = df["App"].apply(lambda x: len(x))
df.iloc[:5,10:] #display attribute 'Name_Length'
#display the unique values in Category
df["Category"].unique()
#Screen the stats for each category
df.groupby('Category').describe()
#remove the outliner within Category of '1.9'
df = df[df.Category != '1.9']
#double check the number of records
df.shape
#display the stats of Rating by Category
df.groupby('Category').Rating.describe()
#fill the missing values of Rating by the mean value of their corresponding app categories
df['Rating'] = df.groupby("Category").Rating.transform(lambda x: x.fillna(x.mean()))
#Double check if any nan value still exist
df.Rating.isna().sum()
#double check the number of records
df.shape
#convert attribute 'Reviews' type into integers
df[['Reviews']] = df[['Reviews']].astype(int)
#display the Reviews's stats and double check the data type
df['Reviews'].describe()
def SizeUnit(str):
    "This function helps to generate a new attribute which specifies the unit of an app's size"
    if str[-1] == 'M':
        return 'M'
    elif str[-1] == 'k':
        return 'k'
    elif str == 'Varies with device':
        return 'Varies with device'
#create a new attribute which specifies the unit of the app's sizes
df['Size_Unit'] = df.Size.apply(lambda x:SizeUnit(x))
#double check if all situations got considered regarding app's sizes
df.Size_Unit.isna().sum()
#remove the unit sympol from "Size" attribute. Note: Now the uncertain value has been written as 'Varies with devic'
df['Size'] = df.Size.apply(lambda x:x[:-1])
df.Size.head()
#replace the uncertain value "Varies with devic" by nan
df['Size'] = df['Size'].replace('Varies with devic', np.nan)
df.head()
#check out the number of nan
df['Size'].isna().sum()
#check out the number of records which have uncertain size, which is consistent with the nan now.
df[df['Size_Unit'] == 'Varies with device'].Size_Unit.count()
#convert the attribute "Size" into float type data
df[['Size']] = df[['Size']].astype(float)
#check out the stats of app's size by category, pick the optimal stat to replace nan
df.groupby('Category').Size.describe()
#check out the number of records
df.shape[0]
#change all value to the same scale. eg. some apps have size value in kb instead of MB
df['Size'] = np.where(df['Size_Unit'] == 'k', df['Size']/1024, df['Size'])
#fill the missing values of Size by the median value of their corresponding app categories
df['Size'] = df.groupby("Category").Size.transform(lambda x: x.fillna(x.median()))
#check out if there's any missing value in attribute 'Size'
df.Size.isna().sum()
#check out the stats of size
df.Size.describe()
#display the first five rows of records
df.head()
#check out number of distict values in attribute 'Intalls'
df.Installs.unique()
#Check out the distribution of Installs
df.groupby('Installs').App.count()
#create a dictionary to record Installs in ascending order
INSTALL = {
    0: '0',1: '0+', 2: '1+',3: '5+',4: '10+', 5: '50+',6: '100+',
    7: '500+',8: '1,000+', 9: '5,000+',10: '10,000+',11: '50,000+', 
    12: '100,000+',13: '500,000+',14: '1,000,000+',15: '5,000,000+', 
    16: '10,000,000+',17: '50,000,000+',18: '100,000,000+', 
    19: '500,000,000+',20: '1,000,000,000+'
}
def NumInstalls(str):
    "find the key of a specifying value in a dictionary"
    for key, value in INSTALL.items():   
        if value == str:
            return key
#create a new attribute to record installs in numerical value
df['Installs_Num'] = df.Installs.apply(lambda x:NumInstalls(x))
#check out if there's any nan value in this new attribute 'Installs_Num'
df.Installs_Num.isna().sum()
#plotting the histogram of installs

fig, ax = plt.subplots(figsize=(12,8))
ax = sns.countplot(x=df['Installs_Num'])
#plt.xticks(rotation=90)
bars = INSTALL.values()
y_pos = np.arange(len(bars))

#rename the xticks with original categorical value
plt.xticks(y_pos, bars, rotation=90, fontsize='13', horizontalalignment='center') 
plt.title("Histogram of Number of APP's Installs", fontsize = '17')
plt.ylabel('Frequency',fontsize = '14')
plt.xlabel('Number of Installs',fontsize = '14')

plt.show()
#create a list with numerical value in terms of installs for future use
Cum_Count = [0, 1, 2, 5, 10, 50, 100, 
         500, 1000, 5000, 10000, 50000, 
         100000, 500000, 1000000, 5000000, 
         10000000, 50000000, 100000000, 500000000, 
         1000000000]
#create a table contains intall frequency and cumulative frequency for plotting purpose
installs_cum = pd.DataFrame(data={'Install': df.groupby('Installs_Num').App.count().index, 
                                  'Freq': df.groupby('Installs_Num').App.count().values})
installs_cum['CumFreq'] = installs_cum['Freq'].cumsum()
installs_cum
#plot the cumulative counts of intalls
sns.set_style('dark')
fig, ax = plt.subplots(figsize=(12,8))
ax = sns.lineplot(x="Install", y="CumFreq", linewidth = '2', color = 'orange', data=installs_cum)

bars2 = ['0', '1', '2', '5', '10', '50', '100', 
         '500', '1,000', '5,000', '10,000', '50,000', 
         '100,000', '500,000', '1,000,000', '5,000,000', 
         '10,000,000', '50,000,000', '100,000,000', '500,000,000', 
         '1,000,000,000']

y_pos = np.arange(len(bars2))
plt.xticks(y_pos, bars2, rotation=90, fontsize='13', horizontalalignment='center')
plt.title("Cumulative Counts of APP's Installs", fontsize = '17')
plt.ylabel('Cumulative Frequency',fontsize = '14')
plt.xlabel('Number of Installs',fontsize = '14')
ax.grid(b=True, which='major')

plt.show()
#test the function of ploty for intalls
installs_cum.CumFreq.iplot()
#convert the attribute 'Price' into numerical value
df['Price'] = np.where(df['Price'] == '0', '$' + df['Price'], df['Price'])
df['Price'] = df.Price.apply(lambda x:x[1:]).astype('float')
#check if there's any nan in attrubute 'Price'
df.Price.isna().sum()
#check out the stats of attribtue 'Price'
df.Price.describe()
#check out the stats of attribtue 'Price' for paid apps only
df[df.Price > 0].Price.describe()
#create a pandas series recording number of counts for each genre
genre = df.groupby('Genres').App.count()
genre.head()
#calculate the apps that belongs to multiple genres
sum = 0
for i in range(len(genre)):
    if ';' in genre.index[i]: #if a string contains ';'
        sum += genre.values[i] #the number of count got added into sum
print(sum)        
#calulate the percentage of records belonging to multiple genres
sum/df.shape[0]
#Keep only one genre for each app
df['Genres'] = df.Genres.apply(lambda x:x.split(';')[0])
#check out if there's any nan in attribute 'Genres'
df.Genres.isna().sum()
#display the number of unique Genres
df.Genres.nunique()
#count number of 'Varies with device'
df[df['Current Ver'] == 'Varies with device'].App.count()
#replace the uncertain value 'Varies with device' with nan
df['Current Ver'] = df['Current Ver'].replace('Varies with device',np.nan)
#double check number of nans
df['Current Ver'].isna().sum()
#remain only the value before '.'
df['test'] = df['Current Ver'].astype('str').apply(lambda x:x.split('.')[0])
df.groupby('test').App.count().head(20)
#delete irrelevant columns
del df['Current Ver']
del df['test']
#drop off irrelevant columns
df = df.drop(columns=['Clean_Name','Size_Unit'])
#display the first five records
df.head()
#check out if there's any nan
df.isna().sum()
#pull out the suspicious record
df[df.Type.isnull()]
#check out what's the typical type for an app in Family type
df[df.Category == 'FAMILY'].groupby('Type').Type.count()
#assign 'Free' as this app's type based on mode
df['Type'] = df.Type.fillna(value='Free')
df.Type.isna().sum() #double check for nan
#display all the attribute's type
df.info()
#display all numerical value's stats
df.describe()
#check out the record with extreme big value in 'Price'
df[df.Price == 400]
#check out the record with extreme big value in 'Name_Length'
df[df.Name_Length == 194].Name_Word_Counts
#check out the record with extreme big value in 'Name_Word_Counts'
df[df.Name_Word_Counts == 21].Name_Length
#save the dataset after preprocessing as a new output csv file
df.to_csv('Clean_googleplaystore.csv')
