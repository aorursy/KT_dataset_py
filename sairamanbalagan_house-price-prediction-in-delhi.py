import pandas as pd                                        #For reading dataframes

import numpy as np                                         #For linear regression

import seaborn as sns                                      #For regression plot and heatmap

from sklearn.model_selection import train_test_split       

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from scipy import stats

%matplotlib inline
df = pd.read_csv('../input/delhi-house-price-prediction/MagicBricks.csv')

df.head()                                                 #Taking a initial look at the dataset
print(df.dtypes)                                         #Understanding the data types of each columns

# We find that Bathroom,Parking has float values which we have to convert to integers.
df.shape                                                #To find how many data is present in this excel
df.describe()
df.isnull().sum()                                       #Finding how many null values in the dataframe
sns.heatmap(df.isnull())

#We can find that there are lots of missing values in per_sqft, thus we are dropping it.

#There are two missing values in bathroom which we can replace with mode function.
df.mode()                                                          #Using mode function to find mode of each variables
df['Type'].value_counts().to_frame()

#Since Builder_Floor has the highest occurances,we are replacing NA type with Builder Floor
df['Furnishing'].value_counts().to_frame()

#Semi furnished gives the highest occurance
sns.heatmap(df.corr(), annot=True)
sns.regplot(x="Parking", y="Price", data=df)

#Parking does not sound like a good predictor of price as data is far from fitted line
#Replacing Bathroom,parking,Type,Furnishing column with their modes:

df['Bathroom'].fillna(value = 2.0, inplace = True)

df['Parking'].fillna(value = 1.0, inplace = True)

df['Type'].fillna(value = "Builder_Floor" , inplace = True)

df['Furnishing'].fillna(value = "Semi-Furnished" , inplace = True)

sns.heatmap(df.isnull())  #Verifying if none of the columns has null values
pearson_coef, p_value = stats.pearsonr(df['Parking'], df['Price'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  

#There is no linear correlation and since the p-value is  >  0.1: there is no evidence that the correlation is significant.
df.drop(['Parking'], axis=1,inplace=True)

df.head()
df.dropna(subset=['Per_Sqft'],axis=0, inplace=True)

df.head()
df.shape
sns.regplot(x="Per_Sqft", y="Price", data=df)
pearson_coef, p_value = stats.pearsonr(df['Per_Sqft'], df['Price'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  

#Since p value is less than 0.01, The correlation is pretty significant between price and per_sqft

#Thus dropping the missing rows is better option than filling it with mean value.
df[['Bathroom']] = df[['Bathroom']].astype("int")

print(df.dtypes)
df.head()                       #Bathroom and parking show int types now.
sns.heatmap(df.isnull()) 

#Finally the data has been cleaned and all the missing values has been removed/replaced.
df['Area'].hist(bins=10)

#We can see that most of the area lies between 0 to 5000
sns.regplot(y="Area", x="Price", data=df)

#Area is a good predictor as data points closely follows the regression plot
pearson_coef, p_value = stats.pearsonr(df['Area'], df['Price'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 

#Since P value is very less, Area is statistically significant althought linear correlation isnt extremely strong
df.sort_values(by='Area', ascending=False).head()

#From the regression plot, the area from 14220.0 seems like an outlier and thus we will remove it for better accuracy
df = df[df.Area < 14220]

df.shape
df['BHK'].hist(bins=10)
sns.regplot(y="BHK", x="Price", data=df)
pearson_coef, p_value = stats.pearsonr(df['BHK'], df['Price'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 

#BHK is statistically significant for predicting price
df['Bathroom'].hist(bins=10)
sns.regplot(y="Bathroom", x="Price", data=df)
pearson_coef, p_value = stats.pearsonr(df['Bathroom'], df['Price'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 

#Bathroom is statistically significant for predicting price because of its low p value and high correlation coefficient
df['Status'].value_counts().to_frame()
df['Locality'].value_counts().to_frame().shape

#We remove locality because it has too many column names
df.drop(['Locality'], axis=1,inplace=True)

df.head()
df['Transaction'].value_counts().to_frame()
df = pd.get_dummies(df)

df.head()
lm = LinearRegression()                                 #Creating a Linear Regression object

lm
x = df[['Area', 'BHK', 'Bathroom', 'Per_Sqft',

       'Furnishing_Furnished', 'Furnishing_Semi-Furnished',

       'Furnishing_Unfurnished', 'Status_Almost_ready', 'Status_Ready_to_move',

       'Transaction_New_Property', 'Transaction_Resale', 'Type_Apartment',

       'Type_Builder_Floor']]

y = df['Price']
from sklearn.model_selection import train_test_split

x_train , x_test , y_train ,y_test = train_test_split(x,y,test_size = 0.3,random_state=0)
linear = LinearRegression()

print(linear.fit(x_train,y_train))

print(linear.score(x_train,y_train))

print(linear.score(x_test,y_test))