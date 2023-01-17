#### Link to Dataset on Kaggle: https://www.kaggle.com/amitranjan01/edmunds-data-analysis-cross-continent-review

    

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import os # accessing directory structure

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#path for kaggle notebook '/kaggle/input'

import os

for dirname, _, filenames in os.walk('../input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# List files available Only run on my local NoteBook

print(os.listdir("../input"))
#Analyze following three Cars from three different region.

#/kaggle/input/edmundsconsumer-car-ratings-and-reviews/Scrapped_Car_Review_Chevrolet.csv

#/kaggle/input/edmundsconsumer-car-ratings-and-reviews/Scraped_Car_Review_mercedes-benz.csv

#/kaggle/input/edmundsconsumer-car-ratings-and-reviews/Scrapped_Car_Reviews_Toyota.csv



# specify 'None' if want to read whole file

nRowsRead = 10000 



# Scrapped_Car_Reviews_Toyota.csv

df_J = pd.read_csv('/kaggle/input/edmundsconsumer-car-ratings-and-reviews/Scrapped_Car_Reviews_Toyota.csv', delimiter=',', nrows = nRowsRead)

#df_J = pd.read_csv('Scrapped_Car_Reviews_Toyota.csv', delimiter=',', nrows = nRowsRead)

df_J.dataframeName = 'Scrapped_Car_Reviews_Toyota.csv'

nRow, nCol = df_J.shape

df_J_orig = df_J.copy(deep=True)

print(f'There are {nRow} rows and {nCol} columns for Toyota')



# Scrapped_Car_Reviews_Toyota.csv

df_E = pd.read_csv('/kaggle/input/edmundsconsumer-car-ratings-and-reviews/Scraped_Car_Review_mercedes-benz.csv', delimiter=',', nrows = nRowsRead)

#df_E = pd.read_csv('Scraped_Car_Review_mercedes-benz.csv', delimiter=',', nrows = nRowsRead)

df_E.dataframeName = 'Scraped_Car_Review_mercedes-benz.csv'

df_E_orig = df_E.copy(deep=True)

nRow1, nCol1 = df_E.shape

print(f'There are {nRow1} rows and {nCol1} columns for Mercedes')



# Scrapped_Car_Reviews_Toyota.csv

df_A = pd.read_csv('/kaggle/input/edmundsconsumer-car-ratings-and-reviews/Scrapped_Car_Review_Chevrolet.csv', delimiter=',', nrows = nRowsRead)

#df_A = pd.read_csv('Scrapped_Car_Review_Chevrolet.csv', delimiter=',', nrows = nRowsRead)

df_A.dataframeName = 'Scrapped_Car_Review_Chevrolet.csv'

df_A_orig = df_A.copy(deep=True)

nRow2, nCol2 = df_A.shape

print(f'There are {nRow2} rows and {nCol2} columns for Chevrolet')
#Analyze the Columns for the datset



print("Columns for Japense Car: ", df_J.columns)

print("Columns for European Car: ", df_E.columns)

print("Columns for American Car: ", df_A.columns)
#Analyze the Columns for the datset



print("Info for Japense Car: ", df_J.info())

print("Info for European Car: ", df_E.info())

print("Info for American Car: ", df_A.info())
#Print head to verify the data

print("Head for Japense Car: ", df_J.head())

print("Head for European Car: ", df_E.head())

print("Head for American Car: ", df_A.head())
print("Data Types for Japense Car: ", df_J.dtypes)

print("Data Types for European Car: ", df_E.dtypes)

print("Data Types for American Car: ", df_A.dtypes)
#Extract the Vechile Make Year from the Vechile Title



df_J['Make_Year'] = df_J['Vehicle_Title'].str[:4]

df_J['Make_Year'] = df_J['Make_Year'].fillna(method ='ffill')

print(df_J['Make_Year'])  



df_E['Make_Year'] = df_E['Vehicle_Title'].str[:4]

df_E['Make_Year'] = df_E['Make_Year'].fillna(method ='ffill')



print(df_E['Make_Year'])  



df_A['Make_Year'] = df_A['Vehicle_Title'].str[:4]

df_A['Make_Year'] = df_A['Make_Year'].fillna(method ='ffill')



print(df_A['Make_Year'])  
# Extract the Review Date from Review

df_J['Review_Date_D'] = df_J['Review_Date'].str[4:12]

df_J['Review_Date_D'] = df_J['Review_Date_D'].fillna(method ='ffill')

df_J['Review_Date_D'] = pd.to_datetime(df_J['Review_Date_D'], errors='coerce')

print(df_J['Review_Date_D'])  



df_E['Review_Date_D'] = df_E['Review_Date'].str[4:13]

df_E['Review_Date_D'] = df_E['Review_Date_D'].fillna(method ='ffill')

df_E['Review_Date_D'] = pd.to_datetime(df_E['Review_Date_D'],  errors='coerce')

print(df_E['Review_Date_D']) 



df_A['Review_Date_D'] = df_A['Review_Date'].str[4:13]

df_A['Review_Date_D'] = df_A['Review_Date_D'].fillna(method ='ffill')

df_A['Review_Date_D'] = pd.to_datetime(df_A['Review_Date_D'], errors='coerce')

print(df_A['Review_Date_D']) 
#Drop data not required for my analysis. In Place update to exisiting Dataframes

df_E.drop(['Unnamed: 0', 'Review_Date', 'Vehicle_Title', 'Review_Title', 'Review'], axis=1, inplace=True)

df_A.drop(['Unnamed: 0', 'Review_Date', 'Vehicle_Title', 'Review_Title', 'Review'], axis=1, inplace=True)

df_J.drop(['Unnamed: 0', 'Review_Date', 'Vehicle_Title', 'Review_Title', 'Review'], axis=1, inplace=True)
print(df_E.info())

print("99th %tile: ", df_E["Rating"].quantile(0.99))

print(df_E.describe())
#Replace the missig entries in Rating column with Mean of Rating

df_E['Rating'].fillna(df_E['Rating'].mean(), inplace=True)

df_A['Rating'].fillna(df_E['Rating'].mean(), inplace=True)

df_J['Rating'].fillna(df_E['Rating'].mean(), inplace=True)
#Replace the missing authoer with anonymous as author

df_E['Author_Name'].fillna('anonymous', inplace=True)

df_A['Author_Name'].fillna('anonymous', inplace=True)

df_J['Author_Name'].fillna('anonymous', inplace=True)
#Add a column representing region where the car belongs. Required to know which data came from which dataframe

df_E['Origin_Region'] = "North America"

df_A['Origin_Region'] = "Eurpoe"

df_J['Origin_Region'] = "Japan"
#After replacing the mean,make sure that the rating is not impacted, we can see 99th %tile:  4.875 is unchanged.

print(df_E.info())

print("99th %tile: ", df_E["Rating"].quantile(0.99))

print(df_E.describe())
print(df_A.info())

print("99th %tile: ", df_A["Rating"].quantile(0.99))

print(df_A.describe())
print(df_J.info())

print("99th %tile: ", df_J["Rating"].quantile(0.99))

print(df_J.describe())
#Append all three dataset

df = df_E.append(df_A).append(df_J)

print(df.shape)

print(df.head(10))

print(df.info())
# convert the floating Rating to Integer

df['Rating'] = df['Rating'].round(0)

df.head()
#Plot the Rating to see how the data is distributed

df['Rating'].plot(kind = 'hist', bins = 100)

plt.show()

#Below Data shows that 5 rating is so frequent that other ratings are not vsisble on graph, 

#so chaging the scale of the graph
#Plot the Rating to see how the data is distributed - Using Log Scale

df['Rating'].plot(kind = 'hist', bins = 100)

plt.yscale('log')

plt.show()
df[df['Origin_Region'] == "North America"]['Rating'].plot(kind = 'hist')

plt.yscale('log')

plt.show()
df[df['Origin_Region'] == "Eurpoe"]['Rating'].plot(kind = 'hist')

plt.yscale('log')

plt.show()
df[df['Origin_Region'] == "Japan"]['Rating'].plot(kind = 'hist')

plt.yscale('log')

plt.show()
plt.figure(figsize=(20,10))

plt.scatter(df['Make_Year'], df['Rating'] )

plt.show()

#Scatter plot doesn't relveal any specific data aspect.
df['Rating'].groupby(df['Author_Name']).count().nlargest(10)

#Anomalies found, top three revewer has reviewed more than 2000 reviews each. 

#Where as the avarage review per reviewe is just 4

#If we drop top reviewes, we will be left with very small set of data. So we will analyze the data in different section.
#Create a Backup copy of data so that we can process it and analyze it later.

df_copy = df.copy(deep=True)
df.drop(df.loc[df['Author_Name']=='anonymous'].index, inplace=True)

df.drop(df.loc[df['Author_Name']=='HD mike'].index, inplace=True)

df.drop(df.loc[df['Author_Name']=='Dave761'].index, inplace=True)

df.drop(df.loc[df['Author_Name']=='Avalon Driver'].index, inplace=True)
#Total dataset is reduced form 30,000 to 348

df.shape
df['Rating'].groupby(df['Author_Name']).count().nlargest(10)
q = df["Rating"].quantile(0.99)

print(q)

df[df["Rating"] < q].count()

#Total of 84 rating are other than 5. Which is about 20% of total left data
df['Rating'].groupby(df['Author_Name']).describe()
df_copy['Rating'].groupby(df_copy['Author_Name']).count().nlargest(10)
#Plot the Rating to see how the data is distributed - Using Log Scale

df['Rating'].plot(kind = 'hist', bins = 100)

plt.yscale('log')

plt.show()

# The new rating distribution seems very similar to what was in original dataframe.
#Now lets analyze the top reviewers rating pattern

print(df_copy[df_copy['Author_Name'] == "Avalon Driver "]['Origin_Region'].count())

df_copy[df_copy['Author_Name'] == "Avalon Driver "]['Origin_Region'].value_counts()

#Seems like Avalon Driver has only reviewd Japanese car and have given all rating of 5
#Now lets analyze the top reviewers rating pattern

print(df_copy[df_copy['Author_Name'] == "Avalon Driver "]['Rating'].count())

df_copy[df_copy['Author_Name'] == "Avalon Driver "]['Rating'].value_counts()

#Seems like Avalon Driver has only reviewd Japanese car and have given all rating of 5
#Now lets analyze the top reviewers rating pattern

print(df_copy[df_copy['Author_Name'] == "Avalon Driver "]['Make_Year'].count())

df_copy[df_copy['Author_Name'] == "Avalon Driver "]['Make_Year'].value_counts()

#Seems like Avalon Driver has given all reviews in one year whcih is 2002
#Now lets analyze the top reviewers rating pattern

print(df_copy[df_copy['Author_Name'] == "Dave761 "]['Origin_Region'].count())

df_copy[df_copy['Author_Name'] == "Dave761 "]['Origin_Region'].value_counts()

#Seems like Avalon Driver has only reviewd Japanese car and have given all rating of 5
#Now lets analyze the top reviewers rating pattern

print(df_copy[df_copy['Author_Name'] == "Dave761 "]['Rating'].count())

df_copy[df_copy['Author_Name'] == "Dave761 "]['Rating'].value_counts()

#Seems like Avalon Driver has only reviewd Japanese car and have given all rating of 5
#Now lets analyze the top reviewers rating pattern

print(df_copy[df_copy['Author_Name'] == "Dave761 "]['Make_Year'].count())

df_copy[df_copy['Author_Name'] == "Dave761 "]['Make_Year'].value_counts()

#Seems like Avalon Driver has given all reviews in one year whcih is 2002
#ax = df_copy.plot(x="Make_Year", y="Rating", kind="bar")

#df_copy.plot(x="Make_Year", y="Rating", kind="bar", ax=ax, color="C2")

#plt.show()
#Load Original Data (unmodified)

print(df_A_orig.info())

print(df_J_orig.info())

print(df_E_orig.info())
df_A_orig1 = df_A_orig.copy(deep=True)

df_A_orig1.dropna(subset = ['Rating'], inplace=True)

print(df_A_orig1.head())

print(df_A_orig1.shape)

df_A_orig1.info()

#If I drop all NA, then I am only left with 51 rows of data out of 10,000 rows. So that is not best option. 

#Lets take another approach
df_E_orig1 = df_E_orig.copy(deep=True)

df_E_orig1.dropna(subset = ['Rating'], inplace=True)

print(df_E_orig1.head())

print(df_E_orig1.shape)

df_E_orig1.info()

#If I drop all NA, then I am only left with 51 rows of data out of 10,000 rows. So that is not best option. 

#Lets take another approach
df_J_orig1 = df_J_orig.copy(deep=True)

df_J_orig1.dropna(subset = ['Rating'], inplace=True)

print(df_J_orig1.head())

print(df_J_orig1.shape)

df_J_orig1.info()

#If I drop all NA, then I am only left with 51 rows of data out of 10,000 rows. So that is not best option. 

#Lets take another approach
#Create new column for Country Of Origin

#Add a column representing region where the car belongs. Required to know which data came from which dataframe

df_E_orig1['Origin_Region'] = "North America"

df_A_orig1['Origin_Region'] = "Eurpoe"

df_J_orig1['Origin_Region'] = "Japan"
# Extract the Review Date from Review

df_J_orig1['Review_Date_D'] = df_J_orig1['Review_Date'].str[4:12]

#df_J_orig1['Review_Date_D'] = df_J_orig1['Review_Date_D'].fillna(method ='ffill')

df_J_orig1['Review_Date_D'] = pd.to_datetime(df_J_orig1['Review_Date_D'], errors='coerce')

print(df_J_orig1['Review_Date_D'])  



df_E_orig1['Review_Date_D'] = df_E_orig1['Review_Date'].str[4:13]

#df_E_orig1['Review_Date_D'] = df_E_orig1['Review_Date_D'].fillna(method ='ffill')

df_E_orig1['Review_Date_D'] = pd.to_datetime(df_E_orig1['Review_Date_D'],  errors='coerce')

print(df_E_orig1['Review_Date_D']) 



df_A_orig1['Review_Date_D'] = df_A_orig1['Review_Date'].str[4:13]

#df_A_orig1['Review_Date_D'] = df_A_orig1['Review_Date_D'].fillna(method ='ffill')

df_A_orig1['Review_Date_D'] = pd.to_datetime(df_A_orig1['Review_Date_D'], errors='coerce')

print(df_A_orig1['Review_Date_D']) 
#Extract the Vechile Make Year from the Vechile Title



df_J_orig1['Make_Year'] = df_J_orig1['Vehicle_Title'].str[:4]

#df_J_orig1['Make_Year'] = df_J_orig1['Make_Year'].fillna(method ='ffill')

print(df_J_orig1['Make_Year'])  



df_E_orig1['Make_Year'] = df_E_orig1['Vehicle_Title'].str[:4]

#df_E_orig1['Make_Year'] = df_E_orig1['Make_Year'].fillna(method ='ffill')

print(df_E_orig1['Make_Year'])  



df_A_orig1['Make_Year'] = df_A_orig1['Vehicle_Title'].str[:4]

#df_A_orig1['Make_Year'] = df_A_orig1['Make_Year'].fillna(method ='ffill')



print(df_A_orig1['Make_Year'])  
df_tot = df_J_orig1.append(df_A_orig1).append(df_E_orig1)

df_tot.info()
#Plot the Rating to see how the data is distributed - Using Log Scale

df_tot['Rating'].plot(kind = 'hist', bins = 15)

plt.yscale('log')

plt.show()

# The new rating distribution seems more like a continous distribution.
df_tot['Rating'].groupby(df_tot['Author_Name']).count().nlargest(10)

#even authors have good reviews distribution with more even review frqquency
df_tot.info()
df_tot_new = df_tot[['Review_Date_D','Author_Name','Rating','Make_Year','Origin_Region']]
df_tot_new.info()
grp = df_tot_new.groupby(['Author_Name','Origin_Region','Make_Year'])['Author_Name','Make_Year','Origin_Region','Rating','Review_Date_D']
grp.describe(include='all')
df_tot_new['Make_Year'] =  pd.to_datetime(df_tot_new['Make_Year'], errors='coerce')
import numpy as np

import matplotlib.pyplot as plt

plt.figure(figsize=(8,12))

#df_tot_new = df_tot[['Review_Date_D','Author_Name','Rating','Make_Year','Origin_Region']]



tw = df_tot_new[df_tot_new['Origin_Region'] == 'Japan']['Make_Year']

tw.sort_values()

tm = df_tot_new[df_tot_new['Origin_Region'] == 'Japan']['Rating']



sw = df_tot_new[df_tot_new['Origin_Region'] == 'Eurpoe']['Make_Year']

sm = df_tot_new[df_tot_new['Origin_Region'] == 'Eurpoe']['Rating']

sw.sort_values()



kw = df_tot_new[df_tot_new['Origin_Region'] == 'North America']['Make_Year']

km = df_tot_new[df_tot_new['Origin_Region'] == 'North America']['Rating']

kw.sort_values()



plt.subplot(3, 1, 1)

plt.scatter(tw, tm)

plt.xlabel('Model Year')

plt.ylabel('Rating')

plt.title('Japan car rating over year')

plt.grid(True)





plt.subplot(3, 1, 2)



plt.scatter(sw, sm)

plt.xlabel('Model Year')

plt.ylabel('Rating')

plt.title('Eurpoe car rating over year')

plt.grid(True)



plt.subplot(3, 1, 3)



plt.scatter(kw, km)

plt.xlabel('Model Year')

plt.ylabel('Rating')

plt.title('United States car rating over year')

plt.grid(True)





plt.tight_layout()

plt.show()
sm.plot()

ticks,labels = plt.xticks()
plt.figure(figsize=(14,8))



X = tw

X1 =sw 

X2 =kw 

Y = tm

Y1 = sm

Y2 = km

plt.scatter(X,Y,  marker = '^', color = 'Green', label ='Japan')

plt.scatter(X1,Y1,  marker = '>', color = 'Red', label ='Eurpoe')

plt.scatter(X2,Y2,  marker = '<', color = 'Blue', label ='North America',)

plt.xlabel('Make Year')

plt.ylabel('Rating')

plt.legend(loc='best')

plt.title('Relationship Between Car make Year and Rating for cars')

plt.ylim(0,6)

plt.show()