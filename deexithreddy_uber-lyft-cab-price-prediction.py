##About the data:



##The data contains 700,000 records of Uber and Lyft data. We try to fit a linear regression model to know the effect of predictors on the price. 



## 

 

#Importing necessary libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



##Importing the csv file

df=pd.read_csv("C:/Users/DEEXITH REDDY/Desktop/Projects/Uber Lyft Linear Regression/rideshare_kaggle.csv")



df=df.drop('timezone', axis=1)  ##Since it contains the same value

df=df.drop('datetime', axis=1)  ##Since this information is stored



df.isnull().sum().sum()/(df.count())*100 ##Since the percentage of na values is less, 7, we can drop without loss of information.



df=df.dropna() ##Dropping na values



##Most picked up and most destinations:



df['source'].describe()         ##Financial district

df['destination'].describe()     ##Financial district





##Since product_id and name give the same information and also name is better described, we drop product_id

df=df.drop('product_id', axis=1)



##Plotting price

plt.hist(df['price'])

##Price is left skewed, but since it is dependent variable, we do not mind. Also, it is continuos.





##Plotting type of taxi

plt.figure(figsize=(10,5))

chart = sns.countplot(

    data=df,

    x='name',

    palette='Set1'

)

chart.set_xticklabels(rotation=45)



##The type of taxi used is more or less the same for each type. Hence, they are equally represented.



##Plotting different type of weather

chart=sns.countplot(x='short_summary',data=df)        

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)



##Plotting greater weather analysis



chart1=sns.countplot(x='long_summary',data=df)        

chart1.set_xticklabels(chart1.get_xticklabels(), rotation=45)



##Plotting source and destination, they were all equal

chart0=sns.countplot(x='source',data=df)        

chart0.set_xticklabels(chart0.get_xticklabels(), rotation=45)

chart2=sns.countplot(x='destination',data=df)        

chart2.set_xticklabels(chart2.get_xticklabels(), rotation=45)



##Plotting time of the day

plt.hist(df['hour'])          ##Spikes at night 00:00, morning 10:00, Afternoon 15:00, night 22:00



##Plotting taxi wise

sns.countplot(x='hour',hue="cab_type",data=df)  ##Not many changes to the hourwise splits for Uber and Lyft





##Plotting daywise

sns.countplot(x='day',hue="cab_type",data=df) ##Less on 9th and 10th of the month



##Plotting month wise

sns.countplot(x='day',hue="month",data=df)  ##End of November to mid of December





##There is a discrepancy between both ICON and SHORT_SUMMARY. We will be deleting the ICON.

##ALso, we will be using long summary, so we will drop even short summary

df=df.drop('icon', axis=1)

df=df.drop('short_summary', axis=1)





##Price is spread out across all times

chart3=sns.scatterplot(x='hour',y='price',data=df)



##We proceed to keep catagorical variables for the source and destination. Hence, we can eliminate the latitude and longitude



df=df.drop('latitude', axis=1)

df=df.drop('longitude', axis=1)





##Separating the price to variable y and dropping of the dataframe



y=df['price']

df=df.drop('price', axis=1)



##Since the timestamp provides all information related to the hour, day and month, we can delete hour, day and month.



df=df.drop(['hour','day','month'], axis=1)



##Setting first column as index:



df=df.set_index(df.columns[0])



##Getting object data type into a single data frame.

obj_df = df.select_dtypes(include=['object']).copy()

obj_df.head()



##Creating dummy variables:



dummy=pd.get_dummies(obj_df, columns=["source", "destination","cab_type","name","long_summary"], prefix=["start", "end","cab","type","weather"])



##Storing the the dataframe so as to first know the effect of non-weather variables and on the price except for weather summary and temperature.



df1=df

df.drop(df.iloc[:, 10:44], inplace = True, axis = 1)

df1=df1.drop('apparentTemperatureMaxTime', axis=1)



##Deleting the object variables since categorical variables have been created.





df1=df1.drop(["source", "destination","cab_type","name","long_summary"], axis=1)





##Making a single dataframe with dummy variables and selected variables



df=pd.concat([df1,dummy], axis=1)



##Splitting into train and test



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df,y,test_size=0.2,random_state=42)



##Checking the mean to compare with MSE value



y.mean()  ##16.54



##Changing all y_train and y_test into lists, to avoid exog, endog error



y_train = list(y_train)

y_test = list(y_test)





##Null Hypothesis: The coefficients are equal to zero

##Alternate Hypothesis: The coefficients are not equal to zero

##Using simple linear regression:



import statsmodels.api as sm

from statsmodels.api import OLS



model = sm.OLS(y_train,x_train).fit()

predictions = model.predict(x_test)



##Evaluating the model

model.summary()



##Model summary:



##As we can see timestamp has high p-value. 

##Distance, surge multiplier, temperature, apparent temperature have low p-values. (Significant)

##Locations start and stop have high p-values and hence are insignificant

##Type of cab also have low value, indicating significance except for Lux SUV

##Cab name also has high p-value, indicating no difference between Uber and Lyft

##All weather variables have high p-values.



##Changes: We remove all the high p-values and use the hour, day, month instead of timestamp

##We recreate the data frame with important variables and add needed dummy variables to the dataframe





df=pd.read_csv("C:/Users/DEEXITH REDDY/Desktop/Projects/Uber Lyft Linear Regression/rideshare_kaggle.csv")

df=df.set_index(df.columns[0])



df.drop(df.iloc[:, 18:55], inplace = True, axis = 1)

df=df.drop(['price','latitude','longitude','apparentTemperatureMaxTime','timestamp','datetime','timezone','source','destination','cab_type','product_id','name'],axis=1)



##Adding the dummy variables created before:



df=pd.concat([df,dummy], axis=1)

df.drop(df.iloc[:, 7:33], inplace = True, axis = 1)

df.drop(df.iloc[:, 19:30], inplace = True, axis = 1)





##Creating training and test sets with the new variables



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df,y,test_size=0.2,random_state=42)



y_train = list(y_train)

y_test = list(y_test)



model = sm.OLS(y_train,x_train).fit()

predictions = model.predict(x_test)



model.summary()



##We come to know that the hour, day and time do not have an effect on the equation

##Now, the temperature and apparent temperature also do not have an effect. 

##So the fare only depends on distance, surge_multiplier and type of cab.



##Deleting the hour, day, month, temperature and apparent temperature

df.drop(df.iloc[:, 0:3], inplace = True, axis = 1)

df=df.drop(['apparentTemperatureMax','temperature '],axis = 1)



##Re-running the model:



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df,y,test_size=0.2,random_state=42)



y_train = list(y_train)

y_test = list(y_test)



model = sm.OLS(y_train,x_train).fit()

predictions = model.predict(x_test)



model.summary()



from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_test, predictions)) ##6.32

np.sqrt(mean_squared_error(y_test, predictions)) ##2.515



##Now we explore different types of linear regression:



##Stochastic Gradient:



from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1)

sgd_reg.fit(x_train, y_train)



predictions = sgd_reg.predict(x_test)



print(mean_squared_error(y_test, predictions))  ##6.41

np.sqrt(mean_squared_error(y_test, predictions)) ##2.53



##The SGD regressor has increase the MSE and RMSE.





##Conclusion:



## With the each increase in unit distance, the price increases by 2.7957, in general

##With each surge multiplier increase, price increases by 18.3234

##The most expensive type is LUX black XL, increasing price by 7.1995

##The least expensive is Uber Pool, decreasing price by -15.6908



##Price does not depend on weather, start and drop off locations or even time

##It only depends on the distance and type of vehicle
