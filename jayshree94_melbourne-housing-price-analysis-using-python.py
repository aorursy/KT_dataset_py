#Load necessary packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Reading the dataset

Housing_MB = pd.read_csv('../input/melbourne-housing-market/Melbourne_housing_FULL.csv')
Housing_MB.head(5)
# exploring and understanding the dataset

print(Housing_MB.shape)

print(Housing_MB.describe())
# Identifying the missing values

Housing_MB.info()
# Understanding variables in Suburb column

print(Housing_MB['Suburb'].value_counts())
# Understanding variables in Type column

print(Housing_MB['Type'].value_counts())

#Most of the houses sold are house,cottage villa or semi terrace type
# Understanding variables in Method column

print(Housing_MB['Method'].value_counts())
print(Housing_MB['CouncilArea'].value_counts())
print(Housing_MB['Regionname'].value_counts())

#Most of the houses sold are from Southern Metropolitan region
print(Housing_MB['Suburb'].value_counts())
# Total Missing value for each feature

print(Housing_MB.isnull().sum())
# Replacing Missing values in columns where we have less than 30% missing values

Housing_MB['Bedroom2'].fillna(Housing_MB['Bedroom2'].median(),axis=0,inplace=True)

Housing_MB['Bathroom'].fillna(Housing_MB['Bathroom'].median(),axis=0,inplace=True)

Housing_MB['Car'].fillna(Housing_MB['Car'].median(),axis=0,inplace=True)

Housing_MB['Landsize'].fillna(Housing_MB['Landsize'].median(),axis=0,inplace=True)

Housing_MB['Lattitude'].fillna(Housing_MB['Lattitude'].median(),axis=0,inplace=True)

Housing_MB['Longtitude'].fillna(Housing_MB['Longtitude'].median(),axis=0,inplace=True)

Housing_MB['Regionname'].fillna(Housing_MB['Regionname'].mode(),axis=0,inplace=True)

Housing_MB['CouncilArea'].fillna(Housing_MB['CouncilArea'].mode(),axis=0,inplace=True)

Housing_MB['Propertycount'].fillna(Housing_MB['Propertycount'].median(),axis=0,inplace=True)
Housing_MB['Regionname'].fillna('Southern Metropolitan',inplace=True)

Housing_MB['CouncilArea'].fillna('Boroondara City Council',inplace=True)
# Validaing the Missing value after missing value is treated for few feature columns

print(Housing_MB.isnull().sum())
Housing_MB['Date']= pd.to_datetime(Housing_MB['Date'],dayfirst=True)
# Grouping the features by Date

var = Housing_MB[Housing_MB['Type']=='h'].sort_values('Date',ascending=False).groupby('Date').std()

count = Housing_MB[Housing_MB['Type']=='h'].sort_values('Date',ascending=False).groupby('Date').count()

mean = Housing_MB[Housing_MB['Type']=='h'].sort_values('Date',ascending=False).groupby('Date').mean()
var
mean
# Average Price marked by varaince by comparing with different date or time when the houses were sold

mean["Price"].plot(yerr=var["Price"],ylim=(400000,1500000))
# Plotting average Landsize marked by variance in price

mean["Landsize"].plot(yerr=var["Price"])
#Group all the features by Date for the houses of type h and Distance less than 14 kms from CBD.

feature_means = Housing_MB[(Housing_MB['Type']=='h')& (Housing_MB['Distance']<14)].sort_values('Date',ascending=False).groupby('Date').mean()

feature_std = Housing_MB[(Housing_MB['Type']=='h') & (Housing_MB['Distance']<14)].sort_values('Date',ascending=False).groupby('Date').std()
#Average no. of Bedroom,Bathroom,Car in Houses sold of h type and which is located within the distance of 14 kms from CBD.

feature_means[['Bedroom2','Bathroom','Car']].plot()
#Average no.of Bedroom,Bathroom,Car marked by variance in Houses sold of h type and which is located within the distance of 14 kms from CBD.

feature_means[['Bedroom2','Bathroom','Car']].plot(yerr=feature_std)
feature_location=Housing_MB[(Housing_MB['Type']=='h')&(Housing_MB['Distance']<14)].sort_values('Date',ascending=False).dropna().groupby(['Suburb']).mean()
#Group all the features by Regionname for the houses of type h and Distance less than 14 kms from CBD.

feature_region_mean=Housing_MB[(Housing_MB['Type']=='h')&(Housing_MB['Distance']<14)].sort_values('Date',ascending=False).dropna().groupby('Regionname').mean()

feature_region_std=Housing_MB[(Housing_MB['Type']=='h')&(Housing_MB['Distance']<14)].sort_values('Date',ascending=False).dropna().groupby('Regionname').std()
# Plotting the avrega eprice of house sold by Regionname

feature_region_mean['Price'].plot(kind='bar',figsize =(15,8))
# Plotting the average no.of Bathroom,Bedroom and Carspots by Regionname

feature_region_mean[['Bedroom2','Bathroom','Car']].plot(yerr=feature_region_std,figsize=(15,8))
# Looking at the average price range in suburb for houses sold in Southern Metropolitan

feature_SouthernM = Housing_MB[(Housing_MB['Regionname']=='Southern Metropolitan')& 

                               (Housing_MB['Type']=='h') & 

                                (Housing_MB['Distance']<=14)].sort_values('Date',ascending=False).groupby('Suburb').mean()
feature_SouthernM['Price'].plot(kind='bar',figsize=(20,10))
#Analyzing Average no. of rooms and Distance for each of the Suburb in Southern Metropolitan Region

feature_South_Suburb = Housing_MB[(Housing_MB['Regionname']=='Southern Metropolitan')& 

                               (Housing_MB['Type']=='h') & 

                                (Housing_MB['Distance']<=14)].sort_values('Date',ascending=False).groupby('Suburb').agg({'Rooms':'median','Distance':'mean'})
feature_South_Suburb
#Analyzing Average no. of rooms and Distance for each of the Suburb in Western Metropolitan Region

feature_West_Suburb = Housing_MB[(Housing_MB['Regionname']=='Western Metropolitan')& 

                               (Housing_MB['Type']=='h') & 

                                (Housing_MB['Distance']<=14)].sort_values('Date',ascending=False).groupby('Suburb').agg({'Rooms':'median','Distance':'mean'})
feature_West_Suburb
# Looking at the average price range in suburb for houses sold in Western Metropolitan

feature_WesternM = Housing_MB[(Housing_MB['Regionname']=='Western Metropolitan')& 

                               (Housing_MB['Type']=='h') & 

                                (Housing_MB['Distance']<=14)].sort_values('Date',ascending=False).groupby('Suburb').mean()
feature_WesternM['Price'].plot(kind='bar',figsize=(20,10))
# Looking at the average price range in suburb for 2 bedroom houses located in the distance of less than 5 kms from CBD sold in Southern Metropolitan 

# Anlyzing the affordable price in the suburbs.

Southern_affordable = Housing_MB[(Housing_MB['Regionname']=='Southern Metropolitan')&

                                (Housing_MB['Rooms']==2)&

                                (Housing_MB['Type']=='h')&

                                (Housing_MB['Distance']<=5)].sort_values('Date',ascending=False).groupby('Suburb').mean()
Southern_affordable['Price'].plot(kind='bar',figsize=(20,10))
# Looking at the average price range in suburb for 2 bedroom houses located in the distance of less than 5 kms from CBD sold in Southern Metropolitan 

# Anlyzing the affordable price in the suburbs.

Western_affordable = Housing_MB[(Housing_MB['Regionname']=='Western Metropolitan')&

                                (Housing_MB['Rooms']==2)&

                                (Housing_MB['Type']=='h')&

                                (Housing_MB['Distance']<=6)].sort_values('Date',ascending=False).groupby('Suburb').mean()
Western_affordable['Price'].plot(kind='bar',figsize=(20,10))
sns.kdeplot(Housing_MB[(Housing_MB['Regionname']=='Southern Metropolitan')

                       &(Housing_MB['Type']=='h')

                       &(Housing_MB['Rooms']==2)]

                       ["Price"])
sns.kdeplot(Housing_MB[(Housing_MB['Regionname']=='Western Metropolitan')

                       &(Housing_MB['Type']=='h')

                       &(Housing_MB['Rooms']==2)]["Price"])
# Plotting the pairplot to understand the distribution and relationship between features

sns.pairplot(Housing_MB.dropna())
# Plotting the heatmap to understand the features correlation

fig,ax = plt.subplots(figsize=(15,15))

sns.heatmap(Housing_MB.corr(),annot=True)
# Plotting the heatmap to understand the features correlation for houses sold of type h

fig,ax = plt.subplots(figsize=(15,15))

sns.heatmap(Housing_MB[Housing_MB['Type']=='h'].corr(),annot=True)
#Drop Null values from dataframe

dataframe_Housing = Housing_MB.dropna().sort_values('Date')
# Convert the date column to number of days from the date when the house is sold

from datetime import date

days_since_start = [(x-dataframe_Housing['Date'].min()).days for x in dataframe_Housing['Date']]

dataframe_Housing['Days']= days_since_start
# Dropping columns which has less correlation to target variable(Price)

df_Housing=dataframe_Housing.drop(['Date','Address','SellerG','Postcode','Landsize','Propertycount'],axis=1)
# understanding the dattyoes from the Housing data frame

df_Housing.dtypes
df_Housing['CouncilArea'].value_counts()
# Convertig Object columns to dummies

df_dummies = pd.get_dummies(df_Housing[['Type','Method','CouncilArea','Regionname']])
df_Housing.columns
#Dropping the old columns which have been converted to dummies and creating a new dataframe

df_Housing.drop(['Suburb','Type','Method','CouncilArea','Regionname'],axis=1,inplace=True)

df_Housing=df_Housing.join(df_dummies)
df_Housing.head(5)
# Splitting indepnedent and dependent features into X and y

from sklearn.model_selection import train_test_split

X= df_Housing.drop(['Price'],axis=1)

y= df_Housing['Price']
# Train test split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=10)
# Train the model

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)
print(lm.intercept_)
lm.score(X_test, y_test)
# Arriving at the coeffecient for the features

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

ranked_suburbs = coeff_df.sort_values("Coefficient", ascending = False)

ranked_suburbs
predictions =lm.predict(X_test)
# Plotting a scatter plot with Predicted and Actual Values based on the trained model

plt.scatter(y_test,predictions)

plt.ylim([200000,1000000])

plt.xlim([200000,1000000])