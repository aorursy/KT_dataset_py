



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

print(os.listdir("../input"))

#Importing the data and forming the inner join between two tables





country_code= pd.read_excel("../input/Country-Code.xlsx")

country_code.head()

print('Shape of country code : ',country_code.shape)



zomato=pd.read_csv('../input/zomato.csv',encoding='latin-1')

zomato.head()

print('Shape of main data : ',zomato.shape)



data=pd.merge(zomato,country_code,how='inner')

print('Shape of data after merging  : ',data.shape)

data.head()
data.columns=['Restaurant_ID', 'Restaurant_Name', 'Country_Code', 'City', 'Address',

       'Locality', 'Locality_Verbose', 'Longitude', 'Latitude', 'Cuisines',

       'Average_Cost_for_two', 'Currency', 'Has_Table_booking',

       'Has_Online_delivery', 'Is_delivering_now', 'Switch_to_order_menu',

       'Price_range', 'Aggregate_rating', 'Rating_color', 'Rating_text',

       'Votes', 'Country']
# Helps in finding the number of null values in the whole dataset



data.isnull().sum() 
#Lets see the records or ros that have null values in the Cuisines features



data[data.Cuisines.isnull()==True]



#The common thing that can be observed from the records belo is that,  all the records with null vaues in the cuisine feature belongs to "United States"

#Lets treat the null values after splitting the data into train ans test data. Becuase, we re however going to dropo the Cusines feature in further code.
data.columns #Gives the number of columns that are in the dataset that we want to work on
data['Switch_to_order_menu'].value_counts() # Gives us the number of unique responses that are in "Switch to order menu " feature
import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'retina'

plt.figure(figsize=(6,4),dpi=100)

data.groupby(['Country']).mean()['Price_range'].sort_values().plot(kind='barh',figsize=(10,6))
#In this plot, number of votes given to a restauatnt on an average can be known.

#We can observe that the votes given are highest for Asian countries(4 out of 5 are Asian) 



%config InlineBackend.figure_format = 'retina'

df=data.groupby(['Country']).mean()

plt.figure(figsize=(8,5),frameon=True,dpi=100)

df['Votes'].sort_values().plot(kind='barh',figsize=(10,6))
data['Rating_color'].value_counts() # Gives us the number of records that has Orange, White, Yellow etc,. in 'Rating Color' feature
data['Rating_text'].value_counts() # Gives us the number of records that has AVergae, Not rated, Good etc,. in 'Rating text' feature
# Features with the following features are droppped :



# That are producing redundant information

# That has 100% percent co relation with other features

# The features that has no significance in predicting the rating

# That requires NLP or other complex algorithms to analyse



data.drop(['Country_Code','Restaurant_ID', 'Restaurant_Name','Address','Locality','Locality_Verbose','Longitude', 'Latitude', 'Switch_to_order_menu','Rating_color'],axis=1,inplace=True)
#Lets rename the feature names without any gaps. Because, gaps in the olumn names may create trouble while indexing.

#It is good practice to avoid spaces, gaps in the column name



data.columns=['City', 'Cuisines', 'Avg_cost', 'Currency',

       'Table_booking', 'Online_delivery', 'Delivering_now',

       'Price_range', 'Rating', 'Rating_text', 'Votes', 'Country']
#lets create 2 dataframes in which one has target variable (i.e. Rating) and latter has predictor variables

# X datframe has predictor variables while y datframe has target variable



X=data.drop('Rating',axis=1)

y=data['Rating']
X.isnull().sum()
#Here, I am splitting 2 dataframes into 4 parts.

#They are 2 target variable datframes (i.e. y_train and y_test) and 2 predictor variables datframes (X_train and X_test)

#X_train is used to train the model using predictor variables while X_test has same features as X_train which is used while testing the model

#y_train is used as a target variable while training the model while y_test is compared with the predicted values after testing the model



#Importing essential library

from sklearn.model_selection import train_test_split





#SPlitting the data into 4 datframes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#Helps in finding the popular cuisines



data.Cuisines.value_counts()
#As the number of cuisines are very high and to make the code identify each cuisines and count it, NLP might be needed.

#So, instead of including the cuisisnes itself, we can also count the number of cuisines offered by the restaurant

# Lets create a new feature that gives the number of cuisines offered by each restaurant



X_train['no_of_cuisines'] = data.Cuisines.str.count(',')+1

X_train.head()
X_test['no_of_cuisines'] = data.Cuisines.str.count(',')+1

X_test.head()
### As there are 9 null values in the cuisisnes, there are also 9 null values in the no_of_cuisines feature.
#In the X_train datframe, lets see the number of records with different number of cuisines



X_train.no_of_cuisines.value_counts()
#In the X_test datframe, lets see the number of records with diferent number of cuisisnes



X_test.no_of_cuisines.value_counts()
#Imputing the null values with the model  in "no_of_cusisnes"



X_train["no_of_cuisines"].fillna(2, inplace = True)

X_test["no_of_cuisines"].fillna(2, inplace = True)
data.isnull().sum()
#A function is being created that helps in assigning continnets to their respective countries



def continent (x):

    if (x in ['United States','Canada','Brazil']):

        return ('Americas')

    elif (x in ['India','Phillipines','Sri Lanka','UAE' ,'Indonesia' ,'Qatar','Singapore']):

        return ('Asia')

    elif (x in ['Australia','New Zealand']):

        return ('Australia_continent')

    elif (x in ['Turkey','United Kingdom']):

        return ('Europe')

    else:

        return ('Africa')
#Here the fuction is being called which creates a new feature named continent by checking with the "Country" feature



X_train['Continent']=X_train['Country'].apply(continent)

X_test['Continent']=X_test['Country'].apply(continent)
#Lets plot the amount spent at restaurants in different countries



%config InlineBackend.figure_format = 'retina'

plt.figure(figsize=(8,5),dpi=100)

data.groupby(['Country']).mean()['Avg_cost'].sort_values().plot(kind='barh',figsize=(10,6))
conversion_rates= {'Botswana Pula(P)':0.095, 'Brazilian Real(R$)':0.266,'Dollar($)':1,'Emirati Diram(AED)':0.272,

    'Indian Rupees(Rs.)':0.014,'Indonesian Rupiah(IDR)':0.00007,'NewZealand($)':0.688,'Pounds(£)':1.314,

    'Qatari Rial(QR)':0.274,'Rand(R)':0.072,'Sri Lankan Rupee(LKR)':0.0055,'Turkish Lira(TL)':0.188}
X_train['New_cost'] = X_train['Avg_cost'] * X_train['Currency'].map(conversion_rates)

X_test['New_cost'] = X_test['Avg_cost'] * X_test['Currency'].map(conversion_rates)
plt.figure(figsize=(8,5),dpi=100)

X_train.groupby(['Country']).mean()['New_cost'].sort_values().plot(kind='barh',figsize=(10,6))
def continent (x):

    if (x in ['United States','Canada','Brazil']):

        return ('Americas')

    elif (x in ['India','Phillipines','Sri Lanka','UAE' ,'Indonesia' ,'Qatar','Singapore']):

        return ('Asia')

    elif (x in ['Australia','New Zealand']):

        return ('Australia_continent')

    elif (x in ['Turkey','United Kingdom']):

        return ('Europe')

    else:

        return ('Africa')
#Here the fuction is being called which creates a new feature named continent by checking with the "Country" feature



X_train['Continent']=X_train['Country'].apply(continent)

X_test['Continent']=X_test['Country'].apply(continent)
#As model can only read numeric values, lets assign values to the rating text, Excellent being the highest(i.e. 5) and poor being the least (i.e. 1)

# These texts will be replaced by the given numbers in train and test data. So that we can include this feature in the model 



dictionary = {'Excellent': 5,'Very Good': 4,'Average': 2,'Good': 3,'Not rated': 2,'Poor': 1} 

X_train.Rating_text = [dictionary[item] for item in X_train.Rating_text] 

X_test.Rating_text = [dictionary[item] for item in X_test.Rating_text] 
#Here encoding is being done in both X_train and X_test dataframes



Binary= {'Yes': 1,'No': 0} 



X_train.Online_delivery = [Binary[item] for item in X_train.Online_delivery] 

X_train.Table_booking = [Binary[item] for item in X_train.Table_booking] 

X_train.Delivering_now = [Binary[item] for item in X_train.Delivering_now] 



X_test.Online_delivery = [Binary[item] for item in X_test.Online_delivery] 

X_test.Table_booking = [Binary[item] for item in X_test.Table_booking] 

X_test.Delivering_now = [Binary[item] for item in X_test.Delivering_now] 
print('Number of cities in the data : ',len(data.City.unique()))
#Lets drop avg_cost feature as new feature is created (i.e. New_cost)

#Cuisines feature is not required as numer of cuisines is created

# Currecny feature is not required as we have standardized everything into dollars

# Lets drop city feature also as there are 141 different cities and when encoding is done, it may create a curse of dimensionality

# If we feel city feature is mandatory, 141 new features will be created and to reduce the dimensions, we need to do PCA

# In this code, as PCA is not being done, lets drop City feature too



X_train.drop(['Avg_cost','Cuisines','Currency','City'],axis=1,inplace=True)

X_test.drop(['Avg_cost','Cuisines','Currency','City'],axis=1,inplace=True)
print(X_train.shape)

print(X_test.shape)
data.head().T
#ENcoding is being done for continents in train and test data sets



train_conti=pd.DataFrame(pd.get_dummies(X_train.Continent))

test_conti=pd.get_dummies(X_test.Continent)
#ENcoding is being done for countries in train and test data sets



train_countr=pd.get_dummies(X_train.Country)

test_countr=pd.get_dummies(X_test.Country)
#The encoded dataframes are being merged to the train and test datasets



X_train=pd.concat([X_train,train_conti,train_countr],axis=1)

X_test=pd.concat([X_test,test_conti,test_countr],axis=1)
X_train.columns
#As country and continent features are included in the datasets in the form of encoded data, lets drop the orginal features



X_train.drop(['Country','Continent'],axis=1,inplace=True)

X_test.drop(['Country','Continent'],axis=1,inplace=True)
X_train.columns
# Lets rename the columns in both train and test datasets



X_train.columns=['Table_booking', 'Online_delivery', 'Delivering_now', 'Price_range',

       'Rating_text', 'Votes', 'no_of_cuisines', 'New_cost', 'Africa',

       'Americas', 'Asia', 'Australia_continent', 'Europe', 'Australia',

       'Brazil', 'Canada', 'India', 'Indonesia', 'NewZealand', 'Phillipines',

       'Qatar', 'Singapore', 'SouthAfrica', 'SriLanka', 'Turkey', 'UAE',

       'UnitedKingdom', 'UnitedStates']

X_test.columns=['Table_booking', 'Online_delivery', 'Delivering_now', 'Price_range',

       'Rating_text', 'Votes', 'no_of_cuisines', 'New_cost', 'Africa',

       'Americas', 'Asia', 'Australia_continent', 'Europe', 'Australia',

       'Brazil', 'Canada', 'India', 'Indonesia', 'NewZealand', 'Phillipines',

       'Qatar', 'Singapore', 'SouthAfrica', 'SriLanka', 'Turkey', 'UAE',

       'UnitedKingdom', 'UnitedStates']
#Importing libraries



from sklearn import model_selection

from scipy.stats import zscore

from sklearn.metrics import explained_variance_score
#Zscore scaling is being done here in both train and test datsets



train_scale=pd.DataFrame(zscore(X_train,axis=1))

test_scale=pd.DataFrame(zscore(X_test,axis=1))
#After scaling the dataset, there will be no feature names. So, lets give feature names



train_scale.columns=['Table_booking', 'Online_delivery', 'Delivering_now', 'Price_range',

       'Rating_text', 'Votes', 'no_of_cuisines', 'New_cost', 'Africa',

       'Americas', 'Asia', 'Australia_continent', 'Europe', 'Australia',

       'Brazil', 'Canada', 'India', 'Indonesia', 'NewZealand', 'Phillipines',

       'Qatar', 'Singapore', 'SouthAfrica', 'SriLanka', 'Turkey', 'UAE',

       'UnitedKingdom', 'UnitedStates']

test_scale.columns=['Table_booking', 'Online_delivery', 'Delivering_now', 'Price_range',

       'Rating_text', 'Votes', 'no_of_cuisines', 'New_cost', 'Africa',

       'Americas', 'Asia', 'Australia_continent', 'Europe', 'Australia',

       'Brazil', 'Canada', 'India', 'Indonesia', 'NewZealand', 'Phillipines',

       'Qatar', 'Singapore', 'SouthAfrica', 'SriLanka', 'Turkey', 'UAE',

       'UnitedKingdom', 'UnitedStates']
#Lets see how the scaling has transformed our original data



train_scale.head()
# Lets round off the values as after certan  nnumber of decimal places, the values does not make significant difference and becomes heavy on the computation



train_scale=np.round(train_scale,decimals=4)

test_scale=np.round(test_scale,decimals=4)

y_train=np.round(y_train,decimals=4)
# implementation of Linear Regression model using scikit-learn





from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.metrics import r2_score
lr = LinearRegression() # Defing the linear regression model 

lr.fit(train_scale,y_train) #Fitting the data into the algorithm

lr_pred = lr.predict(test_scale) #Predicting using Linear regression model



#Metrics for comaprision between prediction and original values

print(r2_score(y_test,np.round(lr_pred,decimals=1))) 

print('RMSE score through Linear regression : ',np.sqrt(metrics.mean_squared_error(y_test,np.round(lr_pred,decimals=1))))
from sklearn.tree import DecisionTreeRegressor

dt= DecisionTreeRegressor()

dt.fit(train_scale,y_train)

dt_pred=dt.predict(test_scale)

print('RMSE score through Decision tree regression : ',np.sqrt(metrics.mean_squared_error(y_test,np.round(dt_pred,decimals=1))))
from sklearn.ensemble import RandomForestRegressor

rf= RandomForestRegressor()

rf.fit(train_scale,y_train)

rf_pred=rf.predict(test_scale)

print('RMSE score through Random Forest : ',np.sqrt(metrics.mean_squared_error(y_test,np.round(rf_pred,decimals=1))))
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4),dpi=100)

plt.plot(y_test,np.round(rf_pred,decimals=1),'*')

plt.xlabel('Actual Rating',size=11)

plt.ylabel('Predicted Ratinge using Random Forest',size=11)

plt.show()
plt.figure(figsize=(6,4),dpi=100)

plt.plot(y_test,rf_pred,'*',label='Random forest')

plt.plot(y_test,dt_pred,'o',color='red',label='Decision tree',marker='s',markersize=4)

plt.legend()

plt.xlabel('Actual VRating',size=11)

plt.ylabel('Predicted Rating',size=11)

plt.show()
import xgboost

xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.3, max_depth=4)

xgb.fit(train_scale,y_train)

xgb_pred= xgb.predict(test_scale)

print('RMSE score through XGBoost : ',np.sqrt(metrics.mean_squared_error(y_test,np.round(xgb_pred,decimals=1))))

print('R square value using XGBoost',r2_score(y_test,xgb_pred))

print('Variance covered by XG Boost Regression : ',explained_variance_score(xgb_pred,y_test))
plt.figure(figsize=(6,4),dpi=100)

plt.plot(y_test,np.round(xgb_pred,decimals=1),'*')

plt.xlabel('Actual Value',size=11)

plt.ylabel('Predicted Value using Random Forest',size=11)

plt.show()
print('RMSE score through Linear Regression : ',np.sqrt(metrics.mean_squared_error(y_test,lr_pred)))

print('R square value using Linear Regression',r2_score(y_test,np.round(lr_pred,decimals=1)))

print('Variance covered by Linear Regression : ',explained_variance_score(lr_pred,y_test))

print('\n')

print('RMSE score through Decision tree Regression : ',np.sqrt(metrics.mean_squared_error(y_test,dt_pred)))

print('R square value using Decision Tree Regression',r2_score(y_test,np.round(dt_pred,decimals=1)))

print('Variance covered by Decision Tree Regression : ',explained_variance_score(dt_pred,y_test))

print('\n')

print('RMSE score through Random Forest : ',np.sqrt(metrics.mean_squared_error(y_test,np.round(rf_pred,decimals=1))))

print('R square value using Random Forest',r2_score(y_test,rf_pred))

print('Variance covered by Random Forest : ',explained_variance_score(rf_pred,y_test))

print('\n')

print('RMSE score through XGBoost : ',np.sqrt(metrics.mean_squared_error(y_test,np.round(xgb_pred,decimals=1))))

print('R square value using XGBoost',r2_score(y_test,xgb_pred))

print('Variance covered by XG Boost Regression : ',explained_variance_score(xgb_pred,y_test))