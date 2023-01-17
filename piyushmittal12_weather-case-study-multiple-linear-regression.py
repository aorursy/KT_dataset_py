# Import Libraries

import numpy as np

import pandas as pd
# Read the dataset

weather1 = pd.read_csv("/kaggle/input/weather-data-for-linear-regression/weather.csv")
#check the head of the data set

weather1.head()
# Inspection code to understand the data

weather1.shape
weather1.info()
weather1.describe()
# Import visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns 



# Pair plot between numerical columns

sns.pairplot(weather1)

plt.show()
# Box plot for Categorical columns

plt.figure(figsize=(20,12))

plt.subplot(2,3,1)

sns.boxplot(x = 'Rain', y = 'Temperature_c' , data = weather1)

plt.subplot(2,3,2)

sns.boxplot(x = 'Description' , y = 'Temperature_c' , data = weather1)



plt.show()
# get the dummy variable for the feature 'Description' and store it in a new variable -'status'

status = pd.get_dummies(weather1['Description'])

#Check what the dataset 'status' looks like 

status.head()
#lets drop the first column from status dataframe using 'drop_first = True'

status = pd.get_dummies(weather1['Description'],drop_first = True)
# Add the result to the original data frame weather1

weather1 = pd.concat([weather1,status],axis = 1)
#Drop 'Description' as we have created the dummies for it

weather1.drop(['Description'],axis=1, inplace= True )
weather1.head()
# Import Library

from sklearn.model_selection import train_test_split



# Split the dataset in 70:30

np.random.seed(0)

df_train, df_test = train_test_split(weather1, train_size = 0.7 , test_size = 0.3 , random_state = 100)

print(df_train.shape)

print(df_test.shape)

# Normalization : (x-xmin)/(xmax - xmin)



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

num_vars1 = ['Humidity','Wind_Speed_kmh','Wind_Bearing_degrees','Visibility_km','Pressure_millibars']

df_train[num_vars1] = scaler.fit_transform(df_train[num_vars1])
df_train.describe()
df_train.head()
plt.figure(figsize = (16,10))

sns.heatmap(df_train.corr(), annot = True , cmap="GnBu")

plt.show()
#Dividing into X and y set for model building



y_train = df_train.pop('Temperature_c')

X_train = df_train
# Building the Model



import statsmodels.api as sm



#Add a constant



X_train_lm = sm.add_constant(X_train[['Humidity']])

# Create a first fitted model

lr  = sm.OLS(y_train, X_train_lm).fit()
lr.params
print(lr.summary())

# Print the summary of the linear regression model obtained

# Assign all the feature variables to X

X_train_lm = X_train[['Humidity','Wind_Speed_kmh'] ]

#Build a linear Model



import statsmodels.api as sm

X_train_lm = sm.add_constant(X_train_lm)

lr  = sm.OLS(y_train, X_train_lm).fit()

lr.params





print(lr.summary())

# Check the summary
X_train_lm = X_train[['Humidity','Wind_Speed_kmh','Wind_Bearing_degrees'] ]

import statsmodels.api as sm

X_train_lm = sm.add_constant(X_train_lm)

lr  = sm.OLS(y_train, X_train_lm).fit()

lr.params
print(lr.summary())

X_train_lm = X_train[['Humidity','Wind_Speed_kmh','Wind_Bearing_degrees','Visibility_km'] ]

import statsmodels.api as sm

X_train_lm = sm.add_constant(X_train_lm)

lr  = sm.OLS(y_train, X_train_lm).fit()

lr.params
print(lr.summary())

X_train_lm = X_train[['Humidity','Wind_Speed_kmh','Wind_Bearing_degrees','Visibility_km','Pressure_millibars'] ]

import statsmodels.api as sm

X_train_lm = sm.add_constant(X_train_lm)

lr  = sm.OLS(y_train, X_train_lm).fit()

lr.params
print(lr.summary())

X_train_lm = X_train[['Humidity','Wind_Speed_kmh','Wind_Bearing_degrees','Visibility_km','Pressure_millibars','Rain'] ]

import statsmodels.api as sm

X_train_lm = sm.add_constant(X_train_lm)

lr  = sm.OLS(y_train, X_train_lm).fit()

lr.params
print(lr.summary())

X_train_lm = X_train[['Humidity','Wind_Speed_kmh','Wind_Bearing_degrees','Visibility_km','Pressure_millibars','Rain','Normal'] ]

import statsmodels.api as sm

X_train_lm = sm.add_constant(X_train_lm)

lr  = sm.OLS(y_train, X_train_lm).fit()

lr.params
print(lr.summary())

X_train_lm = X_train[['Humidity','Wind_Speed_kmh','Wind_Bearing_degrees','Visibility_km','Pressure_millibars','Rain','Normal','Warm'] ]

import statsmodels.api as sm

X_train_lm = sm.add_constant(X_train_lm)

lr  = sm.OLS(y_train, X_train_lm).fit()

lr.params
print(lr.summary())

#Check for the VIF values of the feature variable

from statsmodels.stats.outliers_influence import variance_inflation_factor 
#Create a data frame that will contains all the names of the all the feature variables and their respective VIFs



vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'],2)

vif = vif.sort_values(by = "VIF", ascending = False )

vif
# We generally want a VIF with less than 5 hence we need to drop some variables and consider p also should not be higher than 0.5



#Dropping the variable Pressure_millibars as p is 0.861 as well as VIF very large as compared to 5



X = X_train.drop('Pressure_millibars',1,)
#Build a fitted model again

X_train_lm = sm.add_constant(X)

lr_2  = sm.OLS(y_train, X_train_lm).fit()

lr.params



print(lr.summary())

#Calculate the VIFs again for the new model

vif = pd.DataFrame()

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'],2)

vif = vif.sort_values(by = "VIF", ascending = False )

vif
#High VIF is rain variable with 14.19 large value

X=X.drop('Rain',1)

X_train_lm = sm.add_constant(X)

lr2  = sm.OLS(y_train, X_train_lm).fit()

print(lr2.summary())
#calculate again VIF for new model



#Calculate the VIFs again for the new model

vif = pd.DataFrame()

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'],2)

vif = vif.sort_values(by = "VIF", ascending = False )

vif
#High VIF value hence dropping Visibility_km 7.23



X=X.drop('Visibility_km',1)

X_train_lm = sm.add_constant(X)

lr3  = sm.OLS(y_train, X_train_lm).fit()

print(lr3.summary())




#Calculate the VIFs again for the new model

vif = pd.DataFrame()

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'],2)

vif = vif.sort_values(by = "VIF", ascending = False )

vif
y_train_temp = lr3.predict(X_train_lm)

#Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_temp), bins = 20)

fig.suptitle('Error Terms' , fontsize = 20)

plt.xlabel('Errors',fontsize = 18)                  #X-label  #Plot Heading
