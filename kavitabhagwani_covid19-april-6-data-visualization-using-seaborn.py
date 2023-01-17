# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd





import matplotlib

import matplotlib.pyplot as plt 

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline 
# checking the first 5 rows of dataset

data= pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

data.head()
# Total no of rows and columns 

data.shape
#observation date and last update is in object form so can again import data with datetime object

data.info() 
# Import dataset with date columns in form of datetime object



data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",parse_dates=['ObservationDate','Last Update'])
# can check the total no of daily cases

daily_cases = data.groupby(['ObservationDate'])['Country/Region'].nunique()



# Print head of daily_cases

print(daily_cases.head())
#Visualizing daily cases



# Plot daily_cases

daily_cases.plot()



# Include a title and y-axis label

plt.title('Daily cases')

plt.ylabel('countries')



# Rotate the x-axis labels by 45 degrees

plt.xticks(rotation = 45)



# Display the plot

plt.show()  
# Total No of countries in dataset 

data['Country/Region'].nunique() 
# Convert belows as datatype float to int types  



data['Confirmed']=data['Confirmed'].astype('int')

data['Deaths']=data['Deaths'].astype('int')

data['Recovered']=data['Recovered'].astype('int')
#To check any country confirmed cases like US # Isolate US data 

US = data[data['Country/Region'] == 'US']
# Calculate the total number of confirmed cases in US

total = US['Confirmed'].sum()

print(total)
mortality = US["Deaths"].sum()

print(mortality)
date_arr = data['ObservationDate'].to_numpy()  
confirmed_arr1 = data['Confirmed'].to_numpy()  
death_arr1 = data['Deaths'].to_numpy()  
recover_arr1 = data['Recovered'].to_numpy()  
# Plotting for entire dataset for visualization of confirmed ,death and recovered cases .



# Create a figure with 2x2 subplot layout and make the top left subplot active  # 

fig = plt.figure(figsize=(10,7))

plt.subplot(2, 2, 1) 





# Plot in blue date to confirmed cases 

plt.plot(date_arr,confirmed_arr1, color='blue')

plt.title('confirmed cases')



# Make the top right subplot active in the current 2x2 subplot grid 

plt.subplot(2, 2, 2)



# Plot in red dates to deaths 

plt.plot(date_arr,death_arr1, color='red')

plt.title('death cases')



# Make the bottom left subplot active in the current 2x2 subplot grid

plt.subplot(2, 2,3) 



# Plot in green dates to recovered cases 

plt.plot(date_arr, recover_arr1, color='green')

plt.title('recovered cases')





# Improve the spacing between subplots and display them

plt.tight_layout()

plt.show()
# Compute the maximum confirmed: cs_max

fig = plt.figure(figsize=(7,7))



cs_max = confirmed_arr1.max()



# Calculate the date in which there was maximum cases: date_max

date_max = date_arr[confirmed_arr1.argmax()]



# Plot with legend as before

plt.plot(date_arr, confirmed_arr1, color='red', label='Confirmed') 

plt.plot(date_arr, recover_arr1, color='green', label='recover')

plt.legend(loc='upper left')



# Add a black arrow annotation

plt.annotate('Maximum', xy=(date_max, cs_max), xytext=(date_max+5, cs_max+5), arrowprops=dict(facecolor='black'))



# Add axis labels and title

plt.xlabel('date')

plt.ylabel('cases')

plt.title('confirmed cases vs recovered')

plt.show()
# Generate a 2-D histogram

plt.hist2d(confirmed_arr1,death_arr1,bins=(20, 20),range=((100, 20000), (40, 200)))



# Add a color bar to the histogram

plt.colorbar() 



# Add labels, title, and display the plot

plt.xlabel('confirmed cases')

plt.ylabel('mortality')

plt.title('hist2d() plot')

plt.show()

#Import plotting modules

import matplotlib.pyplot as plt

import seaborn as sns





# Plot a linear regression between 'confirmed' and 'death' cases 

sns.lmplot(x = 'Confirmed', y ='Deaths', data=data)



# Display the plot

plt.show()   



#plot shows a linear correlation btw these two variable
# Generate a green residual plot of the regression between 'confirmed' and 'deaths'

sns.residplot(x='Confirmed', y='Deaths', data=data, color='green')



# Display the plot   # think require transformation seems to be no correlation btw residuals 

plt.show()
# selected only top countries in terms of confirmed cases 



df_1 = data.loc[data['Country/Region'].isin(['US','Spain','Italy','UK','Germany'])]
df_1.shape
# Plot a linear regression between 'confirmed' and 'death', with a hue of 'origin country' and palette of 'Set1' ,only top 5 countries



sns.lmplot(x='Confirmed', y='Deaths', data=df_1, hue='Country/Region',palette='Set1')



# Display the plot

plt.show()

#Make the strip plot again using jitter and a smaller point size to compare the cases 

plt.subplot(2,1,2)

sns.stripplot(x = 'Country/Region', y = 'Confirmed', data = df_1, size = 5, jitter= True)



# Display the plot

plt.show()
# Generate a violin plot of 'countries' grouped horizontally by 'confirmed cases', c 



fig = plt.figure(figsize=(12,10))

plt.subplot(2,1,1)

sns.violinplot(x='Country/Region', y='Confirmed', data= df_1)



# Generate the same violin plot again with a color of 'lightgray' and without inner annotations

plt.subplot(2,1,2)

sns.violinplot(x='Country/Region', y='Confirmed', data=df_1,inner=None,color = 'lightgray')



# Overlay a strip plot on the violin plot

sns.stripplot(x='Country/Region', y='Confirmed', data= df_1, size=3,jitter=True)



# Display the plot

plt.show()
# Can drop SNo col from data as it not much useful

df_1.drop(["SNo"],axis = 1,inplace = True)
#Seaborn's pairplots are an excellent way of visualizing the relationship between all continuous variables in a dataset.



sns.pairplot(df_1)



# Display the plot

plt.show()
# there is strong correlation btwn confirmed cases and deaths 0.86

corr= df_1.corr()

sns.heatmap(corr,annot=True)
# pie chart for to check no of deaths in 5 countries

fig = plt.figure(figsize=(7,7))

conf_per_country = df_1.groupby('Country/Region')['Deaths'].sum().sort_values(ascending=False)

conf_sum= df_1['Deaths'].sum()

def absolute_value(val):

    a  = val

    return (np.round(a,2))

conf_per_country.plot(kind="pie",title='No of deaths per country',autopct=absolute_value)



plt.show ()
