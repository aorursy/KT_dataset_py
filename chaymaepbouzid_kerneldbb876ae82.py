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
#Importing pyplot

from matplotlib import pyplot as plt

from matplotlib import style



#from sklearn.datasets.samples_generator import make_blobs

from sklearn.cluster import KMeans



from pandas import DataFrame



style.use('ggplot')
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error


Toronto_temp = pd.read_csv("../input/toronto-clima/Toronto_temp.csv")

Toronto_temp
Toronto_temp.drop_duplicates(keep='first')

temp_new=Toronto_temp.dropna(axis = 0, how ='any')

temp_new
temp_new.head()
temp_new[:156]
temp_new.Year.head()
temp_new=temp_new.replace('2,018', '2018')

temp_new=temp_new.replace('2,017', '2017')

temp_new=temp_new.replace('2,016', '2016')

temp_new=temp_new.replace('2,015', '2015')

temp_new=temp_new.replace('2,014', '2014')

temp_new=temp_new.replace('2,013', '2013')

temp_new
# Get the summer yearly averages

summer = temp_new.query('season == "Summer"')

summer.rename(columns={'Max Temp (C)':'Summer Temp'}, inplace = True)                      

sum_avg = summer[['Year','Summer Temp']].groupby(['Year'], as_index=False).max()



# Get the winter yearly averages

winter = temp_new.query('season == "Winter"')

winter.rename(columns={'Min Temp (C)':'Winter Temp'}, inplace = True)

win_avg = winter[['Year','Winter Temp']].groupby(['Year'], as_index=False).min()



# Merge the winter and summer averages

year_avg = win_avg

year_avg = year_avg.merge(sum_avg, on= 'Year')



year_avg.plot(subplots=True,kind='line',x='Year', secondary_y=['Summer Temp', 'Winter Temp'],mark_right=False, grid='on',

              title="Toronto Max/Min Temperatures",sharex=True,figsize=(20, 13))



#plt.text(72,21,"Hotest/Coldest temps \nare before 2018 ")

plt.show()
# Get the # Get the first one day from the first month in years

mean_temp = temp_new.query('Month == "1"')

mean_temp = mean_temp.query('Day == "1"')



mean_temp.rename(columns={'Mean Temp (C)':'Mean Temp'}, inplace = True)                      

months_mean = mean_temp[['Year','Month','Day','Mean Temp']]



print(months_mean)



months_mean.plot(x='Year', secondary_y=['Mean Temp'],title="Toronto Max/Min Temperatures",figsize=(20, 13))

plt.show()
# Get the summer yearly temp

summer = temp_new.query('season == "Summer"')

summer.rename(columns={'Max Temp (C)':'Summer Temp Max'}, inplace = True)                      

sum_avg_max = summer[['Year','Summer Temp Max']].groupby(['Year'], as_index=False).max()



summer.rename(columns={'Min Temp (C)':'Summer Temp Min'}, inplace = True) 

sum_avg_min = summer[['Year','Summer Temp Min']].groupby(['Year'], as_index=False).min()



plt.plot(sum_avg_max['Year'],sum_avg_max['Summer Temp Max'],linewidth=1)

plt.plot(sum_avg_min['Year'],sum_avg_min['Summer Temp Min'],linewidth=1)



plt.title('Temp Min/Max')

plt.ylabel('Temp')

plt.xlabel('Year')



plt.legend()



plt.grid(True)



plt.show()
Winter = temp_new.query('season == "Winter"')

Winter.rename(columns={'Max Temp (C)':'Winter Temp Max'}, inplace = True)                      

win_avg_max = Winter[['Year','Winter Temp Max']].groupby(['Year'], as_index=False).max()



Winter.rename(columns={'Min Temp (C)':'Winter Temp Min'}, inplace = True) 

win_avg_min = Winter[['Year','Winter Temp Min']].groupby(['Year'], as_index=False).min()



plt.plot(win_avg_max['Year'],win_avg_max['Winter Temp Max'],linewidth=1)

plt.plot(win_avg_min['Year'],win_avg_min['Winter Temp Min'],linewidth=1)



plt.title('Temp Min/Max')

plt.ylabel('Temp')

plt.xlabel('Year')



plt.legend()



plt.grid(True)



plt.show()
# Get the Spring yearly averages

summer = temp_new.query('season == "Spring"')

summer.rename(columns={'Total Rain (mm)':'Spring Rain'}, inplace = True)                      

sum_avg = summer[['Year','Spring Rain']].groupby(['Year'], as_index=False).max()



# Get the Fall yearly averages

winter = temp_new.query('season == "Fall"')

winter.rename(columns={'Total Rain (mm)':'Fall Rain'}, inplace = True)

win_avg = winter[['Year','Fall Rain']].groupby(['Year'], as_index=False).min()



# Merge the Fall and summer averages



year_avg = win_avg

year_avg = year_avg.merge(sum_avg, on= 'Year')



year_avg.plot(subplots=True,kind='line',x='Year', secondary_y=['Spring Rain', 'Fall Rain'],mark_right=False, grid='on',

              title="Toronto Rain Temperatures",sharex=True,figsize=(20, 5))



plt.text(68,20,"Rain \nare before 2012 ")

plt.show()
spring_rain = temp_new.query('season == "Spring"')

spring_rain.rename(columns={'Total Rain (mm)':'Spring Rain'}, inplace = True)                      

spring_rain_year = spring_rain[['Year','Spring Rain']].groupby(['Year'], as_index=False).sum()



summer_rain = temp_new.query('season == "Summer"')

summer_rain.rename(columns={'Total Rain (mm)':'Summer Rain'}, inplace = True)                      

summer_rain_year = summer_rain[['Year','Summer Rain']].groupby(['Year'], as_index=False).sum()



fall_rain = temp_new.query('season == "Fall"')

fall_rain.rename(columns={'Total Rain (mm)':'Fall Rain'}, inplace = True)                      

fall_rain_year = fall_rain[['Year','Fall Rain']].groupby(['Year'], as_index=False).sum()



winter_rain = temp_new.query('season == "Winter"')

winter_rain.rename(columns={'Total Rain (mm)':'Winter Rain'}, inplace = True)                      

winter_rain_year = winter_rain[['Year','Winter Rain']].groupby(['Year'], as_index=False).sum()





plt.plot(spring_rain_year['Year'],spring_rain_year['Spring Rain'],linewidth=1)

plt.plot(summer_rain_year['Year'],summer_rain_year['Summer Rain'],linewidth=1)



plt.plot(fall_rain_year['Year'],fall_rain_year['Fall Rain'],linewidth=1)

plt.plot(winter_rain_year['Year'],winter_rain_year['Winter Rain'],linewidth=1)





plt.title('Total Rain')

plt.ylabel('Rain')

plt.xlabel('Year')



plt.legend()



plt.grid(True)



plt.show()
spring_rain = temp_new.query('season == "Spring"')

spring_rain.rename(columns={'Total Snow (cm)':'Spring Snow'}, inplace = True)                      

spring_rain_year = spring_rain[['Year','Spring Snow']].groupby(['Year'], as_index=False).sum()



summer_rain = temp_new.query('season == "Summer"')

summer_rain.rename(columns={'Total Snow (cm)':'Summer Snow'}, inplace = True)                      

summer_rain_year = summer_rain[['Year','Summer Snow']].groupby(['Year'], as_index=False).sum()



fall_rain = temp_new.query('season == "Fall"')

fall_rain.rename(columns={'Total Snow (cm)':'Fall Snow'}, inplace = True)                      

fall_rain_year = fall_rain[['Year','Fall Snow']].groupby(['Year'], as_index=False).sum()



winter_rain = temp_new.query('season == "Winter"')

winter_rain.rename(columns={'Total Snow (cm)':'Winter Snow'}, inplace = True)                      

winter_rain_year = winter_rain[['Year','Winter Snow']].groupby(['Year'], as_index=False).sum()



plt.plot(winter_rain_year['Year'],winter_rain_year['Winter Snow'],linewidth=1)

plt.plot(spring_rain_year['Year'],spring_rain_year['Spring Snow'],linewidth=1)

plt.plot(fall_rain_year['Year'],fall_rain_year['Fall Snow'],linewidth=1)

plt.plot(summer_rain_year['Year'],summer_rain_year['Summer Snow'],linewidth=1)



plt.title('Total Snow')

plt.ylabel('Snow')

plt.xlabel('Year')



plt.legend()



plt.grid(True)



plt.show()
# Get the Winter yearly averages

winter = temp_new.query('season == "Winter"')

winter.rename(columns={'Total Rain (mm)':'Winter Rain'}, inplace = True)                      

sum_avg = winter[['Year','Winter Rain']].groupby(['Year'], as_index=False).max()



# Get the Fall yearly averages

winter = temp_new.query('season == "Winter"')

winter.rename(columns={'Total Snow (cm)':'Winter Snow'}, inplace = True)

win_avg = winter[['Year','Winter Snow']].groupby(['Year'], as_index=False).max()



# Merge the Fall and summer averages

year_avg = win_avg

year_avg = year_avg.merge(sum_avg, on= 'Year')



year_avg.plot(subplots=True,kind='line',x='Year', secondary_y=['Winter Rain', 'Fall Rain'],mark_right=False, grid='on',

              title="Toronto Rain Temperatures",sharex=True,figsize=(20, 10))



plt.show()
# Get the first month in 2018

year_rain = temp_new.query('Month == "1"')

#year_rain = years_rain.query('Day == "1"')

year_rain.rename(columns={'Total Rain (mm)':'Total Rain'}, inplace = True)                      

months_rain = year_rain[['Year','Month','Day','Mean Temp (C)','Total Rain','Total Snow (cm)']]



plt.bar(months_rain['Year'], months_rain['Total Rain'], align='center')



plt.bar(months_rain['Year'], months_rain['Total Snow (cm)'], color='g', align='center')



plt.title('Rain/Snow Info')

plt.ylabel('Y')

plt.xlabel('S/N')



plt.show()
#Make a copy of DF



df_tr = temp_new[['Year','Month','Day','Mean Temp (C)','Total Rain (mm)','Total Snow (cm)']]

print(df_tr)

#Transsform the timeOfDay to dummies

#Year	Month	Day	Mean Temp (C)	Max Temp (C)	Min Temp (C)	Total Rain (mm)	Total Snow (cm)

# from sklearn.cluster import KMeans 

clusters = 7

  

kmeans = KMeans(n_clusters = clusters) 

kmeans.fit(df_tr) 

  

print(kmeans.labels_)
import seaborn as sns 

  

# generating correlation heatmap 

sns.heatmap(df_tr.corr(), annot = True) 

  

# posting correlation heatmap to output console  

plt.show()


Error =[]

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i).fit(df_tr)

    kmeans.fit(df_tr)

    Error.append(kmeans.inertia_)

import matplotlib.pyplot as plt

plt.plot(range(1, 11), Error)

plt.title('Elbow method')

plt.xlabel('No of clusters')

plt.ylabel('Error')

plt.show()


#df = temp_new[['Mean Temp (C)','Total Rain (mm)','Total Snow (cm)']]

df_rain = temp_new[['Mean Temp (C)','Total Rain (mm)']]

np.random.seed(200)

k = 3

# centroids[i] = [x, y]

centroids = {

    i+1: [np.random.randint(0, 80), np.random.randint(0, 80)]

     # Measure the distance to every center

    for i in range(k)

}

    

fig = plt.figure(figsize=(5, 5))

plt.scatter(df_rain['Mean Temp (C)'], df_rain['Total Rain (mm)'], color='k')

colmap = {1: 'r', 2: 'g', 3: 'b'}

 # Calculate mean for every cluster and update the center

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

plt.xlim(0, 80)

plt.ylim(0, 80)

plt.show()
#df = temp_new[['Mean Temp (C)','Total Rain (mm)','Total Snow (cm)']]

df_snow = temp_new[['Mean Temp (C)','Total Snow (cm)']]

np.random.seed(200)

k = 3

# centroids[i] = [x, y]

centroids = {

    i+1: [np.random.randint(0, 80), np.random.randint(0, 80)]

    for i in range(k)

}

    

fig = plt.figure(figsize=(5, 5))

plt.scatter(df_snow['Mean Temp (C)'], df_snow['Total Snow (cm)'], color='k')

colmap = {1: 'r', 2: 'g', 3: 'b'}

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

plt.xlim(0, 80)

plt.ylim(0, 80)

plt.show()
#print(year_rain)

year_rain = year_rain.query('Year == "2018"')

test_year_rain=year_rain



fig = plt.figure(figsize=(7,15))



ax1 = fig.add_subplot(3,1,1)

ax2 = fig.add_subplot(3,1,2)



year_rain.plot(x='Total Rain', y='Mean Temp (C)', ax=ax1, kind='scatter')

year_rain.plot(x='Total Snow (cm)', y='Mean Temp (C)', ax=ax2, kind='scatter')



plt.show()
lr = LinearRegression()

lr.fit(year_rain[['Total Rain']], year_rain['Mean Temp (C)'])



print(lr.coef_)

print(lr.intercept_)



a0 = lr.intercept_

a1 = lr.coef_
lr.fit(year_rain[['Total Rain']], year_rain['Mean Temp (C)'])



rain_predictions = lr.predict(year_rain[['Total Rain']])

test_predictions = lr.predict(test_year_rain[['Total Rain']])



rain_mse = mean_squared_error(rain_predictions, year_rain['Mean Temp (C)'])

test_mse = mean_squared_error(test_predictions, test_year_rain['Mean Temp (C)'])



rain_rmse = np.sqrt(rain_mse)

test_rmse = np.sqrt(test_mse)



print(rain_rmse)

print(test_rmse)
cols = ['Total Rain', 'Mean Temp (C)']

lr.fit(year_rain[cols], year_rain['Mean Temp (C)'])



rain_predictions = lr.predict(year_rain[cols])

test_predictions = lr.predict(test_year_rain[cols])



rain_rmse_2 = np.sqrt(mean_squared_error(rain_predictions, year_rain['Mean Temp (C)']))

test_rmse_2 = np.sqrt(mean_squared_error(test_predictions, test_year_rain['Mean Temp (C)']))



print(rain_rmse_2)

print(test_rmse_2)
temp_new[:50]
df = DataFrame(temp_new,columns=['Year','Month','Day','Mean Temp (C)','Total Rain (mm)','Total Snow (cm)'])



plt.scatter(df['Total Rain (mm)'], df['Mean Temp (C)'], color='red')

plt.title('Mean Temp Vs Total Rain', fontsize=14)

plt.xlabel('Total Rain', fontsize=14)

plt.ylabel('Mean Temp', fontsize=14)

plt.grid(True)

plt.show()

 

plt.scatter(df['Total Snow (cm)'], df['Mean Temp (C)'], color='green')

plt.title('Mean Temp Vs Total Snow', fontsize=14)

plt.xlabel('Total Snow', fontsize=14)

plt.ylabel('Mean Temp', fontsize=14)

plt.grid(True)

plt.show()
from sklearn import linear_model

import statsmodels.api as sm



X = df[['Total Rain (mm)','Total Snow (cm)']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets

Y = df['Mean Temp (C)']

 

# with sklearn

regr = linear_model.LinearRegression()

regr.fit(X, Y)



print('Intercept: \n', regr.intercept_)

print('Coefficients: \n', regr.coef_)



# prediction with sklearn

New_Mean_Temp = 5.0

New_Total_Rain = 2.0

print ('Predicted Mean Temp (C): \n', regr.predict([[New_Mean_Temp ,New_Total_Rain]]))



# with statsmodels

X = sm.add_constant(X) # adding a constant

 

model = sm.OLS(Y, X).fit()

predictions = model.predict(X) 

 

print_model = model.summary()

print(print_model)