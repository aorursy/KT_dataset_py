# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # Easy plotting

import matplotlib.pyplot as plt # Base plotting 

import pandasql as pdsql #To run SQL quries on pandas DataFrame



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
ramen_data = pd.read_csv('/kaggle/input/ramen-ratings/ramen-ratings.csv')
ramen_data.head()
#As Review# is more like a serial number we will drop it 

ramen_data.drop(columns = 'Review #',inplace = True)
ramen_data.shape
ramen_data['Top Ten'].isnull().value_counts()
#In Top ten column there is position along with year.We can use that.

ramen_data['Top Ten'].value_counts().head()
top10_data = ramen_data[ramen_data['Top Ten'].notna()]

ramen_data.drop(columns = 'Top Ten',inplace = True)
#As we can see in top10_data that we have extracted previously

#have some typing mistakes

top10_data[top10_data['Top Ten'] == '\n']
top10_data =top10_data[top10_data['Top Ten'] != '\n']
#Now lets check for the null values

ramen_data.isnull().sum()
#Lets drop thes 2 value records

ramen_data.dropna(inplace = True)

ramen_data.isnull().sum()
ramen_data.shape
#Lets covert the datatype of Stars column to integer 

ramen_data.dtypes
ramen_data['Stars'].value_counts()
#There are 3 unrated ramen types so we have handle these

#before we change the data type of column Stars to flaot

ramen_data['Stars'].value_counts()['Unrated']

ramen_data = ramen_data[ramen_data['Stars'] != 'Unrated']
#Here we are converting the data type of Stars column from Object to Float

ramen_data['Stars'] = ramen_data['Stars'].astype(float)

ramen_data.dtypes
#Here there are two records for one for USA and another for United States

#which are same

ramen_data.Country.value_counts()
ramen_data[(ramen_data['Country'] =='United States') | (ramen_data['Country'] =='USA')]['Country'].value_counts()
#We will be adding the United States record to USA by chnaging the country name

ramen_data['Country'].replace('United States','USA',inplace =True)
ramen_data[(ramen_data['Country'] =='United States') | (ramen_data['Country'] =='USA')]['Country'].value_counts()
#Lets do some visualization

ramen_data.head()
#Lets see which country has given most ramen reviews

review_data = ramen_data.groupby(by = 'Country').count().sort_values(by = 'Stars',ascending =False)

plt.figure(figsize = (10,10))

sns.barplot(data = review_data,y=review_data.index ,x= 'Stars',orient='h' )

plt.xlabel('Number of reviews')

plt.title('Country Vs No. of reviews')

#Most number of reviews were given by Japan followed by South Koera and USA

#and it also shows were ramen is most popular JAPAN
#Lets check top 10 brand which has most number variety in ramen

vairety_data = pdsql.sqldf("SELECT Brand,count(Variety) as Number_Of_Variety FROM ramen_data GROUP BY Brand ")

vairety_data.sort_values(by = 'Number_Of_Variety',ascending =False,inplace =True)
#Using bar plot and pie chart

fig = plt.figure(figsize = (14,6))

fig.suptitle('Brand Vs Number Of Variety')



#Using barplot from seaborn

ax1 = fig.add_subplot(121)

sns.barplot(data = vairety_data.head(10),y='Brand' ,x= 'Number_Of_Variety',orient='h',ax =ax1)

plt.xlabel('Number Of Variety')



#Using pie chart from matplotlib

ax2 = fig.add_subplot(122)

plt.pie(vairety_data.head(10)['Number_Of_Variety'],

        explode = (0.1,0,0,0,0,0,0,0,0,0),labels =vairety_data.head(10)['Brand'],

        autopct='%1.1f%%',

        shadow =True)

# Nissin has the most number of variety around 400
#Prima Taste was in 1st position 5 times

fig = plt.figure(figsize = (14,5))

data =pd.DataFrame(top10_data.groupby(by='Brand')['Top Ten'].count())

data.sort_values(by = 'Top Ten',ascending =False,inplace = True)

sns.barplot(data=data, y='Top Ten', x=data.index,orient ='v')

plt.xticks(rotation=45)