# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import needed libraries



import pandas as pd

import matplotlib.pyplot as plt



#library with regression, 'from' commands let you pick specific functions 

from sklearn.linear_model  import LinearRegression

from sklearn.model_selection import train_test_split 

from sklearn import metrics



#another regression library 

import seaborn as seabornInstance 



#name data

lib_rain = pd.read_csv("../input/lib_rain.csv")



#check head of dataset 

lib_rain.head()
# looks like I should have removed the indexing when I exported the CSV

# oh well! I will just remove the extra column 



lib_rain = lib_rain [['year_month', 'Checkouts', 'average_rain']]



#look at data to see that it workd

lib_rain.head()
# need to grab x and y values to give to linear regressor

# the library needs it as a numpy array 

# request values and save in numpy array 

# reshape returns a list of arguments and puts it into x 

# -1 shows we only want one row out of it



y = lib_rain.iloc[:,1].values.reshape(-1,1)

x = lib_rain.iloc[:,2].values.reshape(-1,1)



plt.scatter(x,y,s =10)



plt.title('Rainfall as a predictor for library item checkout volume')

plt.xlabel('average inches of rain')

plt.ylabel('number of items checked out')



# need prediction line

model = LinearRegression()



# use x and y to fit the line

model.fit(x,y)



#hold y's predicition in the variable 

#y is being predicted by the value of x 

y_pred = model.predict(x)



plt.scatter(x,y,s =10)



#plot the predicition line

plt.plot(x,y_pred, color ='red')
# but how good is this predicition? 

# I can test this out by comparing actual to predicted.

# I got this idea from: https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f



y = lib_rain.iloc[:,1].values.reshape(-1,1)

x = lib_rain.iloc[:,2].values.reshape(-1,1)



#split the data so 80% of it is training data and 20% of it is testd ata 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)



regressor = LinearRegression() 



#training the algorithm

regressor.fit(x_train, y_train)



# set up predicitions for the test data

y_pred = regressor.predict(x_test)



#create a new df to compare the actual data to predicted data

df_rain_items = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})



df_rain_items.head()
#create a bar graph that compares actual number of books to predicted number of books for the first 24 items



df1 = df_rain_items.head(25)

df1.plot(kind='bar',figsize=(10,8))

plt.show()



#plot the test data on a scatter plot for an aditional visual confirmation

plt.scatter(x_test, y_test)

plt.plot(x_test, y_pred, color='red', linewidth=2)
# Accuracy measure shows us the % accuaracy of our model's prediction 

model.score(x_test,y_test)
# I am curious to see if date has a big impact

# we can't take the regression of a string so I need to convert the year_month data to datetime data

# and then store it as ordinal data in a new column

# I can use the date_ordinal data when calculating the regression



lib_rain['date_ordinal'] = pd.to_datetime(lib_rain ['year_month']).apply(lambda date: date.toordinal())



# see if that worked

lib_rain.head()
# Now I will create a regression model to predict number of items checked out by date



y = lib_rain.iloc[:,1].values.reshape(-1,1)

x = lib_rain.iloc[:,3].values.reshape(-1,1)



#plot the x and y values & set figure size

plt.figure(figsize=(40,10))

plt.scatter(x,y)



#create plot labels and change font sizes so it is easier to read

plt.title('Library Checkouts Over Time, April 2005 to May 2017', fontsize=35)

plt.xlabel('Year & Month',fontsize=30)

plt.ylabel('Number of Items Checked Out', fontsize=30)



#set x axis tick positions to ordianl date data and x axis tick labels to year_month data

#this way we can plot the data, but still label it in a way that is easy to read

positions = (lib_rain['date_ordinal'])

labels = (lib_rain['year_month'])

plt.xticks(positions, labels)



#rotate x ticks for easier reading

plt.xticks(rotation=45)



#set the limits for x and y ticks so the plot has less white space

#these ordinal numbers were obtained from looking at the dataframe

plt.xlim((732000,736500))



# create a prediction line

model = LinearRegression()



# use x and y to fit the line

model.fit(x,y)



#hold y's predicition in the variable 

#y is being predicted by the value of x 

y_pred = model.predict(x)



#plot the predicition line in red so it stands out

plt.plot(x,y_pred, color ='red')
# but again, I wonder, how good is this predicition? 

# Just as before, I can test this out by comparing actual to predicted



y = lib_rain.iloc[:,1].values.reshape(-1,1)

x = lib_rain.iloc[:,3].values.reshape(-1,1)



#split the data so 80% of it is training data and 20% of it is testd ata 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)



regressor = LinearRegression() 



#training the algorithm

regressor.fit(x_train, y_train)



# set up predicitions for the test data

y_pred = regressor.predict(x_test)



#create a new df to compare the actual data to predicted data

df_date_items = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})



df_date_items.head()
#create a bar graph of the first 24 items to compare the predicted amount to the actual amount



df2 = df_date_items.head(25)

df2.plot(kind='bar',figsize=(10,8))

plt.show()



#plot the test data on a scatter plot for an aditional visual confirmation

plt.scatter(x_test, y_test)

plt.plot(x_test, y_pred, color='red', linewidth=2)



#create plot labels and change font sizes so it is easier to read

plt.title('Library Checkouts Over Time - Test Data', fontsize=15)

plt.xlabel('Year & Month (ordinal)',fontsize=10)

plt.ylabel('Number of Items Checked Out', fontsize=10)



plt.show()
# Accuracy measure

model.score(x_test,y_test)
# declare x and y values. In this case we have two x values

X = lib_rain[['average_rain','date_ordinal']].values

y = lib_rain['Checkouts'].values
# next declare variables to help us train and test the model

# 20% of our data will be used to test the model, 80% will be used to train it. 



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# set up the model



model = LinearRegression()  

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



#Check the difference between the actual value and predicted value. This might give more insight into what is going on

#It also helps me see that the model is working properly

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df.head()
# check the model accuracy measure

model.score(X_test,y_test)
#plot actual vs predicted number of checkouts to see this accuracy in a more visual way 



df.plot(kind='bar',figsize=(10,8))

plt.show()
lib_rain_old = lib_rain [['average_rain','date_ordinal','Checkouts']]
# we can look at pearson correlations to get a better sense of the data

# a number closer to 1 indicates a stronger correlation

# This is why for example, 'Checkouts' correlate 1.000 with 'Checkouts'

# becuase they are always the same variable 



lib_rain_old.corr()
# I can also look at these correlations visually 

# lighter colors such as yellow and blue-greenindicate a strong correlation 



plt.matshow(lib_rain_old.corr())

plt.xticks(range(len(lib_rain_old.columns)), lib_rain_old.columns)

plt.yticks(range(len(lib_rain_old.columns)), lib_rain_old.columns)

plt.colorbar()

plt.show()
# here is another way to visualize those correlations

pd.plotting.scatter_matrix(lib_rain_old, figsize=(8, 8))

plt.show()
# my last assingnment it was suggested I try and plot

# checkouts (a discrete value) as bars, with avg rain plotted as a line over them.

# I am interested in how this works in python so I am going to give that a try as well





lib_rain['Checkouts'].plot(kind='bar', figsize=(25,10))

lib_rain['average_rain'].plot(secondary_y=True, color='red', linewidth=2)





#create plot labels and change font sizes so it is easier to read

plt.title('Library Checkouts and Rainfall Over Time, April 2005 to May 2017', fontsize=35)

plt.xlabel('Year & Month')

plt.ylabel('Inches of rain', fontsize=30)
