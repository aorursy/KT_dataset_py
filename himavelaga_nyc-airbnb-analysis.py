

import numpy as np

import pandas as pd

import io

import os

!pip install pydotplus 



df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
# drop unwanted columns and null values

d = df.drop(['name', 'host_id', 'reviews_per_month', 'host_name', 'latitude', 'longitude', 'last_review', 'calculated_host_listings_count' ], axis=1)

d = d.dropna(axis =0)
import matplotlib.pyplot as plt



#Statistics for number of reviews

print("Number of Reviews statistics: ")

print(d.number_of_reviews.describe())

print("Median: ", d.number_of_reviews.median())



#Price range distribution for each neighborhood

bins = [0,50, 100, 150, 200, 250, 300, 400, 500, 600, 10000]

groups2 = d.groupby(['neighbourhood_group', pd.cut(d.price, bins)])

groups2 =  groups2.size().unstack()

groups2 = groups2.fillna(0).T

groups2.plot.bar(figsize=(10,15), title='Number of airbnbs within each price range within each neighborhood group')

plt.xlabel('Price ranges')

plt.ylabel('Number of Airbnbs')



# Number of reviews for each room type for each neighborpood group

max1 = d['number_of_reviews'].max()

min1 = d['number_of_reviews'].min()

x = ([0, 50, 100,200,300,400,500,600,700])

split1 = pd.cut(d.number_of_reviews, x)

counts1 = d.groupby(['neighbourhood_group', 'room_type'])['number_of_reviews'].mean().to_frame()

counts1 = pd.pivot_table(counts1, index='neighbourhood_group', columns='room_type', values='number_of_reviews')

counts1.plot.bar(figsize=(10,15), title='Average Number of reviews for each room type for each neighborhood group', color=['r', 'g', 'b', 'k', 'm', 'y'])

plt.xlabel('Neighboorhood group')

plt.ylabel('Average Number of reviews')



#Top 30 airbnbs and their average price, availability and popular room type and neighborhood group

print("\nTop 30 airbnb listings and their characteristics: ")

largest = d.nlargest(30, 'number_of_reviews')['price']

largest2 = d.nlargest(30, 'number_of_reviews')['availability_365']

largest3 = d.nlargest(30, 'number_of_reviews')['room_type']

largest4 = d.nlargest(30, 'number_of_reviews')['neighbourhood_group']

print('Average price: ', largest.mean())

print('Average Availability: ', largest2.mean())

x = largest3.value_counts()

x2 = x.idxmax()

print('Room type: ', x.loc[[x2]])

x3 = largest4.value_counts()

x4 = x3.idxmax()

print('Neighborhood: ', x3.loc[[x4]])



#Availability vs Number of Reviews

ax1 = d.plot(kind='scatter', x='availability_365', y='number_of_reviews', color='r')

ax1.set_xlabel("Availability 365")

ax1.set_ylabel('Number of reviews')

ax1.set_title("Availability 365 Days vs {}".format('Number of reviews'))

import numpy as np

from sklearn.preprocessing import PolynomialFeatures



##GRAPH-Number of reviews for each neighborhood  -> popularity of each neighbourhood in each of the 5 boroughs

  #-> get the top 5 areas in each neighborhood(brooklyn, manhattan...)

fig, (ax, ax1, ax2, ax3, ax7) = plt.subplots(1, 5, sharey=True, figsize=(15,7))



#make dataframes for each of the five boroughs so that we can analyze each of their neighbourhoods

df1 = d[d['neighbourhood_group'] == 'Manhattan'].reset_index()

df2 = d[d['neighbourhood_group'] == 'Brooklyn'].reset_index()

df3 = d[d['neighbourhood_group'] == 'Queens'].reset_index()

df4 = d[d['neighbourhood_group'] == 'Bronx'].reset_index()

df5 = d[d['neighbourhood_group'] == 'Staten Island'].reset_index()



#groupby all the neighbourhoods by each neighbourhood_group/borough and get the average of their number of reviews. From these averages find the top 5 neighbourhoods with the

#highest average and plot a bar graph 

group=df1.groupby('neighbourhood')['number_of_reviews'].mean().nlargest(5).plot.bar(ax=ax, color=['r','b','b','b','b'])

group2=df2.groupby('neighbourhood')['number_of_reviews'].mean().nlargest(5).plot.bar(ax=ax1, color=['r','b','b','b','b'])

group3=df3.groupby('neighbourhood')['number_of_reviews'].mean().nlargest(5).plot.bar(ax=ax2, color=['r','b','b','b','b'])

group4=df4.groupby('neighbourhood')['number_of_reviews'].mean().nlargest(5).plot.bar(ax=ax3, color=['r','b','b','b','b'])

group5=df5.groupby('neighbourhood')['number_of_reviews'].mean().nlargest(5).plot.bar(ax=ax7, color=['r','b','b','b','b'])



#Give all the plots x labels and a y label as well as a title

fig.suptitle('Average Number of Reviews for top 5 Neighbourhoods', fontsize=14)

ax.set_xlabel('Manhattan')

ax1.set_xlabel('Brooklyn')

ax2.set_xlabel('Queens')

ax3.set_xlabel('Bronx')

ax7.set_xlabel('Staten Island')

ax.set_ylabel('Average Number of Reviews')



##GRAPH-Average price for each room type



#get the price for each room type into array

arr=d.loc[d['room_type'] == "Entire home/apt", 'price'].reset_index()

arr2=d.loc[d['room_type'] == "Private room", 'price'].reset_index()

arr3=d.loc[d['room_type'] == "Shared room", 'price'].reset_index()



#got the average of each array

homeavg=(np.mean(arr['price']))

privavg=(np.mean(arr2['price']))

shareavg=(np.mean(arr3['price']))



averages= []

house=['Entire home/apt', 'Private room', 'Shared room']

averages.append(homeavg)

averages.append(privavg)

averages.append(shareavg)



#plotted the averages price for room type

fig1, ax4 = plt.subplots()

ax4.set_title('Average Price per Room Type')

ax4.set_xlabel('Room Type')

ax4.set_ylabel('Average Price')

ax4.bar(house, averages, color=['r','b','g'])





## GRAPH-Which price ranges have heavy number of bookings(most number of reviews)?

fig, ax5 = plt.subplots(figsize=(5,5))

ax5.scatter(d['price'], d['number_of_reviews']) #positively skewed

ax5.set_xlim(1,500)

ax5.set_xlabel("Price Range")

ax5.set_ylabel("Number of Reviews")

ax5.set_title("Prices with Heaviest Number of Reviews")



## GRAPH-which price ranges have more minimum nights stayed?

fig2, ax6 = plt.subplots(figsize=(5,5))

ax6.scatter(d['price'], d['minimum_nights'], color='g')

ax6.set_xlim(1,500)

ax6.set_ylim(0,500)

ax6.set_xlabel("Price Range")

ax6.set_ylabel("Minimum Nights Stayed")

ax6.set_title("Prices with Heaviest Minimum Nights Stayed")
#********************************************************ANALYSIS WITH MACHINE LEARNING*********************************************************

import seaborn as sns

from sklearn.metrics import mean_squared_error 

from sklearn.preprocessing import PolynomialFeatures

import numpy as np



from scipy.optimize import leastsq

import scipy as sc

import matplotlib.pyplot as plt



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score





#CHANGE PRICE TO NORMAL DISTRIBUTION USING LOG

d = d[d.price > 0]

d['price'] = np.log(d.price + 1)

X = d['price'].values.reshape(-1,1)



#POLYNOMIAL TRANSFORMATION OF PRICE FEATURE

polynomial_features= PolynomialFeatures(degree=2)

X_poly = polynomial_features.fit_transform(X)

y = d['number_of_reviews'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=0)

#LINEAR REGRESSION

regressor = LinearRegression()  

regressor.fit(X_train, y_train) #training the algorithm

y_pred = regressor.predict(X_test)

m = mean_squared_error(y_test, y_pred)

r = r2_score(y_test, y_pred)

xtest2 = X_test[:,-1]

plt.scatter(xtest2, y_test,  color='gray')

plt.plot(xtest2, y_pred, color='red', linewidth=2)

#plt.figure(figsize=(20,10))

plt.xlabel('Price') 

plt.ylabel('Number of Reviews')

plt.title("Polynomial Regression of price vs number of reviews")

print('MSE of Price vs Number of Reviews', m)

print('R2 score of Price vs Number of Reviews', r, '\n')





#Simple linear regression for analyzing room type vs number of reviews

X_cat = d['room_type']

X_cat = pd.get_dummies(data=X_cat, drop_first=True)

y = d['number_of_reviews']

X_train, X_test, y_train, y_test = train_test_split(X_cat, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  

regressor.fit(X_train, y_train) #training the algorithm

y_pred = regressor.predict(X_test)

m = mean_squared_error(y_test, y_pred)

r = r2_score(y_test, y_pred)

print('MSE of Room type vs Number of Reviews: ', m) #

print('R2 score of Room type vs Number of Reviews: ', r, '\n')



#Simple linear regression for analyzing neighborhood group vs number of reviews

X_l = d['neighbourhood_group']

X_l = pd.get_dummies(data=X_l, drop_first=True)

y = d['number_of_reviews']

X_train, X_test, y_train, y_test = train_test_split(X_l, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  

regressor.fit(X_train, y_train) #training the algorithm

y_pred = regressor.predict(X_test)

m = mean_squared_error(y_test, y_pred)

r = r2_score(y_test, y_pred)

print('MSE of Location vs Number of Reviews: ', m) #

print('R2 score of Location vs Number of Reviews: ', r, '\n')





#Multiple linear regression with features price, room type and neighborhood group

df = d[[ 'price']]

X0 = pd.concat([df, X_l], axis = 1)

X1 = pd.concat([X0, X_cat], axis = 1)

y = d['number_of_reviews']

X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  

regressor.fit(X_train, y_train) #training the algorithm

y_pred = regressor.predict(X_test)

m = mean_squared_error(y_test, y_pred)

r = r2_score(y_test, y_pred)

print('MSE of Multiple Regression(Room type, location & Price vs Number of Reviews): ', m)

print('R2 score of Multiple Regression(Room type, location & Price vs Number of Reviews): ', r, '\n')



XX = pd.concat([X_l, X_cat], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(XX, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  

regressor.fit(X_train, y_train) #training the algorithm

y_pred = regressor.predict(X_test)

m = mean_squared_error(y_test, y_pred)

r = r2_score(y_test, y_pred)

print('MSE of Multiple Regression(Room type & location vs Number of Reviews): ', m)

print('R2 score of Multiple Regression(Room type & location vs Number of Reviews): ', r, '\n')





# print('MAE (Mean Absolute Error): %s' %mae)

# print('MSE (Mean Squared Error): %s' %mse)

# print('RMSE (Root mean squared error): %s' %rmse)

# print('R2 score: %s' %r2)
#Decision tree

from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



#Decided that everything equal to or below 5 reviews would be considered a low amount of reviews and everything above would be a high amount of reviews

# 5 was chosen since it was the median of the number of reviews 

d['label'] = np.where((d['number_of_reviews'] >= 0) & (d['number_of_reviews']<= 5), '0', '1')



#One hot encoding for room_type

X_room = d['room_type']

X_room = pd.get_dummies(data=X_room, drop_first=True)

#One hot encoding for neighborhood group

X_n = d['neighbourhood_group']

X_n = pd.get_dummies(data=X_n, drop_first=True)

X_rn = pd.concat([X_n, X_room], axis = 1)

#cut price into bins to make it more categorical

df = d[[ 'price']]

bins = [0,50, 100, 150, 200, 250, 300, 400, 500, 600, 10000]

groups2 =  pd.cut(d.price, bins)

groups2 = pd.get_dummies(data=groups2, drop_first=True)

#price to intervals to one hot encoding

X1 = pd.concat([X_rn, groups2], axis = 1)



#Train and test data by 80% and 20% split respectively

y = d['label']

X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=101)

model = tree.DecisionTreeClassifier()

m = model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print("Accuracy Score: ",accuracy_score(y_test, y_predict), "\n")



#Confusion matrix

con = pd.DataFrame(confusion_matrix(y_test, y_predict), columns=['Predicted Low # of Reviews', 'Predicted High # of Reviews'], index=['True Low  # of Reviews', 'True High # of Reviews'])

print("Confusion Matrix \n", con, "\n")

print("Classification Report \n", classification_report(y_test, y_predict), "\n")



#Display decision tree





from sklearn.externals.six import StringIO  

from IPython.display import Image

from sklearn.tree import export_graphviz

import pydotplus



dot_data = StringIO()

export_graphviz(model, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())