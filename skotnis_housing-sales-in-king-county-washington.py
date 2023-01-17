import numpy as np

import pandas as pd

import statsmodels.api as sm

import matplotlib.pyplot as plt

import seaborn as sns



#override the matplotlib style of graphs wiht seaborn.

sns.set()
raw_data = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")

raw_data.head()
raw_data.info()
raw_data.describe(include='all')

#We observe no missing values at first which we confirm later. 

#There seem to be a lot of house with some that are exceptionally priced.
raw_data.isnull().sum()
data = raw_data.drop(['id','date'],axis = 1)

data.head()
#change the size of the figure using this matplotlib me.thod

plt.subplots(figsize=(15,10))



#plot a correalation heatmap using seaborn. Border the squares with black color, show the correaltion index and round it up.

sns.heatmap(data.corr(), annot = True,linewidths=.5,linecolor='black',fmt = '1.1f')



#give a title to the map and display it.

plt.title('correlation heatmap',size = 18)

plt.show()
#We use this in-built seaborn method to plot the specified variables and display regression lines to summarize the trends.

sns.pairplot(data,vars = ["price","bedrooms","bathrooms","floors"], kind ="reg")
#pandas mehtod to obtain the unique values of this variable to understand which values have been taken.

data['bedrooms'].unique()
plt.subplots(figsize=(12,10))



#seaborn method to plot a boxplot using the specified variables from the dataset.

sns.boxplot(x="bedrooms", y = "price",data= data)



plt.title('price vs no of bedrooms',size = 18)

plt.show()
data['bathrooms'].unique()
plt.subplots(figsize=(12,10))



#The underscore is a dummy variable used for making it 2D.

_=plt.hist(data['bathrooms'],color='salmon')

_=plt.xlabel('no of bathrooms')

_=plt.ylabel('price')



plt.title('price vs no of bathrooms',size = 18)

plt.show()
data['floors'].unique()
plt.subplots(figsize=(15,10))



#seaborn method for plotting a bargraph.

sns.barplot(x="floors",y="price",data=data,palette="Blues_d")



plt.title('price vs no of floors',size = 18)

plt.show()


#seaborn method for a 'relpot':view acts as a further breakdown dimension. We change the look of the graph by using another color pallate.

sns.relplot(x="sqft_living",y="price",hue="waterfront",col="view",palette=["g", "r"],data=data)



sns.pairplot(data, vars = ["sqft_living","sqft_lot","sqft_basement","sqft_living15","sqft_lot15"], kind ="reg")
data['grade'].unique()
plt.subplots(figsize=(12,10))



sns.boxplot(x="grade", y = "price",data= data,palette="Set3")



plt.title('price vs grade',size = 18)

plt.show()
data['condition'].unique()
plt.subplots(figsize=(12,10))



sns.boxplot(x="condition", y = "price",data= data,palette="Set1")



plt.title('price vs condition',size = 18)

plt.show()
plt.subplots(figsize=(15,10))



#we use a scatterplot to analyze the relationship between price and grade and further break it down using condiiton.

sns.scatterplot(x="grade",y="price",hue="condition",size="condition",sizes=(20, 200),data=data)



plt.title('relationship between price,grade and condition',size = 18)

plt.show()
plt.subplots(figsize=(15,15))



#this matplotlib method gives us the distribution of the counts of the first 50 observations and the year in which they were built. 

#the argument passed in displays the percentage upto the first decimal place.

data.yr_built.value_counts().head(50).plot.pie(autopct='%1.1f%%')



plt.title('year built pie chart',size = 18)

plt.show()
data.yr_built.value_counts()
plt.subplots(figsize=(12,10))



plt.scatter(data['long'],data['lat'],color="purple")



#we set the limits according to the cartographical convention.

plt.xlim(-180,180)

plt.ylim(-180,180)



plt.xlabel('longitude')

plt.ylabel('latitude')



plt.title('distribution of houses',size = 18)

plt.show()
#we zoom in for a better picture.



plt.subplots(figsize=(12,10))



plt.scatter(data['long'],data['lat'],color="purple")



#note that the coordinates have been selected based on the output of the previous scatterplot and hence won't be 100% accurate.

plt.xlim(-121.2,-122.6)

plt.ylim(47,47.9)



plt.xlabel('longitude')

plt.ylabel('latitude')



plt.title('distribution of houses closeup',size = 18)

plt.show()
plt.subplots(figsize=(12,10))



y=data['price']

x=data['sqft_living']



plt.scatter(x,y,color='green')



plt.title('price vs living area',size = 18)

plt.show()

plt.show()
plt.subplots(figsize=(8,8))



#this in-built seaborn method plot the necessary graph.

sns.distplot(data['price'],color='crimson')



plt.title('pdf of price',size = 18)

plt.show()
plt.subplots(figsize=(8,8))



#we create a new variable to contain the observations in the 99th percentile, that is the most dramatic outliers.

q = data['price'].quantile(0.99)



#we store it in a new data fram that contains all the observations except for the top 1 percentile. They would normally represent some luxury houses.

data_1 = data[data['price']<q]

data_1.describe(include = "all")



sns.distplot(data_1['price'],color='crimson')



plt.title('pdf of price less than the 99th percentile',size = 18)

plt.show()
plt.subplots(figsize=(8,8))



sns.distplot(data_1['bedrooms'],color='m')



plt.title('pdf of no of bedrooms',size = 18)

plt.show()
plt.subplots(figsize=(8,8))



p=data_1['bedrooms'].quantile(0.99)

data_2 = data_1[data_1['bedrooms']<p]



sns.distplot(data_2['bedrooms'],color='m')



plt.title('pdf of no of bedrooms less than 99th percentile',size = 18)

plt.show()
plt.subplots(figsize=(12,10))



sns.distplot(data_2['sqft_living'],color='pink')



plt.title('pdf of living area',size = 18)

plt.show()
plt.subplots(figsize=(8,8))

p = data_2['sqft_living'].quantile(0.99)



data_3 = data_2[data_2['sqft_living']<p]



sns.distplot(data_3['sqft_living'],color='pink')



plt.title('pdf of price less than 99th percentile',size = 18)

plt.show()
plt.subplots(figsize=(8,8))



sns.distplot(data_2['sqft_basement'],color='cyan')



plt.title('pdf of basement',size = 18)



plt.show()
plt.subplots(figsize=(8,8))



p = data_3['sqft_basement'].quantile(0.99)

data_4 = data_3[data_3['sqft_basement']<p]



sns.distplot(data_4['sqft_basement'],color='cyan')



plt.title('pdf of basement with 99th percentile',size = 18)

plt.show()
data_model=data_3.reset_index(drop=True)

data_model.describe(include="all")
#we can directly use the numpy method to convert all the price datapoints and store it a new variable.

log_price=np.log(data_model['price'])



#we store the new variable in a new column in the existing dataframe.

data_model['log_price'] = log_price



data_model.head()
plt.subplots(figsize=(8,8))



y=data_model['log_price']

x=data_model['sqft_living']



plt.scatter(x,y,color='green')



plt.title('log price vs living area',size = 18)

plt.show()
data_model.columns.values
#please note that this method has been directly used as per the statsmodels documentation. There is no inbuilt method to calculate the vif but the algorithm is cited and cab be found in the accompanying documentation.

from statsmodels.stats.outliers_influence import variance_inflation_factor



variables = data_model[[ 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot','floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above','sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat','long', 'sqft_living15', 'sqft_lot15']]



vif = pd.DataFrame()



vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]

vif["Features"] = variables.columns
vif
data_cleaned = data_model.drop(['sqft_living15','sqft_lot15','sqft_above','sqft_lot','sqft_basement',],axis = 1)

data_cleaned.head()
data_cleaned.columns.values
#In the above equation, x1 is x or the values taken by the x variable and y is the values taken by the y variable.



x1=data_cleaned[[ 'bedrooms', 'bathrooms', 'sqft_living',

       'floors', 'waterfront', 'view', 'condition', 'grade',

        'yr_built', 'yr_renovated', 'zipcode', 'lat',

       'long']]



y=data_cleaned[['log_price']]

#again, this method is pre-existing and can be directly used. The citation can be found in the documentation.



#this is b0. We are essentially adding a coulmn consisting of only 1s that is equal in length to the y variable.

x= sm.add_constant(x1)



#we fit the regression model on x and y using the appropriate method and store it in a variable.

results = sm.OLS(y,x).fit()



#we summarize our findings.

results.summary()



#note that there is a variable to represent the error in the image. In statistical terms it is the SSE. In easier words, we are trying to minimize the error. The lower the error, the better our model is.
data_reg=data_model.drop(['long','yr_renovated'],axis=1)

data_reg.head()
x1=data_cleaned[[ 'bedrooms', 'bathrooms', 'sqft_living',

       'floors', 'waterfront', 'view', 'condition', 'grade',

        'yr_built', 'lat','zipcode']]

y=data_cleaned[['log_price']]
x= sm.add_constant(x1)

results = sm.OLS(y,x).fit()

results.summary()
#the first column displays the constant that we added earlier.

x

#we create a new dataframe with some observations.

data_with_predictions = pd.DataFrame({'const':1,'bedrooms':[3,3],'bathrooms':[1,2.25],'sqft_living':[1180,2570],'floors':[1,2],'waterfront':[0,0],'view':[0,0],'condition':[3,3], 'grade':[7,7],'yr_built':[1955,1951],'lat':[47.5112,47.7210],'zipcode':[98103,98002]})



#we name the columns and display it.

data_with_predictions=data_with_predictions[['const','bedrooms','bathrooms','sqft_living','floors','waterfront','view','condition','grade','yr_built','lat','zipcode']]



data_with_predictions
predictions = results.predict(data_with_predictions)

predictions
#we store the predictions in a new variable and attach it to the dataframe

data_with_predictions['predictions'] = predictions

data_with_predictions
#using the inbuilt numpy method we take the exponent(inverse of logarthim)of the logarithmic price to get the original prices that we are interested in.

pred_price=np.exp(data_with_predictions['predictions'])



pred_price
#again we store it in a new variable and attach it to the dataset.

data_with_predictions['predicted_price']=pred_price

data_with_predictions