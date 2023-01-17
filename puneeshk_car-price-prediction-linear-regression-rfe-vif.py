# Supress Warnings

import warnings

warnings.filterwarnings('ignore')
import numpy as np

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



import matplotlib.pyplot as plt

import seaborn as sns



import statsmodels

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor



import sklearn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
# Read the csv

carprice = pd.read_csv("../input/car-price-dataset/CarPrice_Assignment.csv")
# Read the head

carprice.head()
# 205 rows and 26 columns

carprice.shape
carprice.info()
carprice.describe()
# Checking Null values%

round(100*(carprice.isnull().sum()/len(carprice.index)),2)

# There are no NULL values
# Drop the car_ID column as to does not hold any significance for developing the model

carprice.drop(['car_ID'], axis = 1, inplace = True)
# Convert CarName to lower string

carprice['CarName'] = carprice['CarName'].str.lower()



# Create a new column called company from the first word in CarName Values

carprice['company'] = carprice['CarName'].str.split(' ').str[0]



print(carprice['company'].value_counts())
# Perform corrections in the company names

carprice['company'].replace(to_replace="vokswagen", value = 'volkswagen', inplace=True)

carprice['company'].replace(to_replace="vw", value = 'volkswagen', inplace=True)

carprice['company'].replace(to_replace="toyouta", value = 'toyota', inplace=True)

carprice['company'].replace(to_replace="porcshce", value = 'porsche', inplace=True)

carprice['company'].replace(to_replace="maxda", value = 'mazda', inplace=True)



print(carprice['company'].value_counts())
# Drop the CarName column now as we have created a new column of company which will be used in analysis and modeling

carprice.drop(['CarName'], axis = 1, inplace = True)
print(carprice['enginelocation'].value_counts(),"\n")
# Almost all the values of enginelocation are front; hence dropping that column

carprice.drop(['enginelocation'], axis = 1, inplace = True)
plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

plt.title('Car Price Distribution Plot')

sns.distplot(carprice.price)



plt.subplot(1,2,2)

plt.title('Car Price Spread')

sns.boxplot(y=carprice.price)



plt.show()

carprice.price.describe()
print(carprice['company'].value_counts(),"\n")

print(carprice.groupby("company").price.mean().sort_values(ascending=False))



plt.figure(figsize=(16, 8))



plt.subplot(2,1,1)

ax1 = sns.countplot(y="company", data = carprice)

ax1.set(ylabel='Car Company', xlabel='Count of Cars')



plt.subplot(2,1,2)

ax2 = sns.barplot(y="company", x = "price" , data = carprice)

ax2.set(ylabel='Car Company', xlabel='Average Car Price')



plt.show()
print(carprice['symboling'].value_counts(),"\n")

print(carprice.groupby("symboling").price.mean())



plt.figure(figsize=(12, 6))



plt.subplot(1,2,1)

ax1 = sns.countplot(x="symboling", data = carprice)

ax1.set(xlabel='Insurance Risk Rating', ylabel='Count of Cars')



plt.subplot(1,2,2)

ax2 = sns.barplot(x="symboling", y = "price" , data = carprice)

ax2.set(xlabel='Insurance Risk Rating', ylabel='Average Car Price')



plt.show()
print(carprice['fueltype'].value_counts(),"\n")

print(carprice.groupby("fueltype").price.mean())



plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)

ax1 = sns.countplot(x="fueltype", data = carprice)

ax1.set(xlabel='Type of Fuel', ylabel='Count of Cars')

plt.subplot(1,2,2)

ax2 = sns.barplot(x="fueltype", y = "price" , data = carprice)

ax2.set(xlabel='Type of Fuel', ylabel='Average Car Price')

plt.show()
print(carprice['aspiration'].value_counts(),"\n")

print(carprice.groupby("aspiration").price.mean())

plt.figure(figsize=(10, 5))

ax = sns.barplot(x="aspiration", y = "price" , data = carprice)

ax.set(xlabel='Aspiration', ylabel='Average Car Price')

plt.show()
print(carprice['doornumber'].value_counts(),"\n")

print(carprice.groupby("doornumber").price.mean())



plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)

ax1 = sns.countplot(x="doornumber", data = carprice)

ax1.set(xlabel='Number of Doors', ylabel='Count of Cars')

plt.subplot(1,2,2)

ax2 = sns.barplot(x="doornumber", y = "price" , data = carprice)

ax2.set(xlabel='Number of Doors', ylabel='Average Car Price')

plt.show()
print(carprice['carbody'].value_counts(),"\n")

print(carprice.groupby("carbody").price.describe())



plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)

ax1 = sns.countplot(x="carbody", data = carprice)

ax1.set(xlabel='Car Body', ylabel='Count of Cars')

plt.subplot(1,2,2)

ax2 = sns.barplot(x="carbody", y = "price" , data = carprice)

ax2.set(xlabel='Car Body', ylabel='Average Car Price')

plt.show()
# Based upon data visualization of car body, replace hardtop and convertible values to a single value

carprice['carbody'].replace(to_replace="hardtop", value = 'hardtop_or_convertible', inplace=True)

carprice['carbody'].replace(to_replace="convertible", value = 'hardtop_or_convertible', inplace=True)

print(carprice['carbody'].value_counts(),"\n")
print(carprice['cylindernumber'].value_counts(),"\n")

print(carprice.groupby("cylindernumber").price.mean().sort_values(ascending=False))



plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)

ax1 = sns.countplot(x="cylindernumber", data = carprice)

ax1.set(xlabel='Number of cylinders', ylabel='Count of Cars')

plt.subplot(1,2,2)

ax2 = sns.barplot(x="cylindernumber", y = "price" , data = carprice)

ax2.set(xlabel='Number of cylinders', ylabel='Average Car Price')

plt.show()
print(carprice['enginetype'].value_counts(),"\n")

print(carprice.groupby("enginetype").price.mean().sort_values(ascending=False))



plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)

ax1 = sns.countplot(x="enginetype", data = carprice)

ax1.set(xlabel='Engine Type', ylabel='Count of Cars')

plt.subplot(1,2,2)

ax2 = sns.barplot(x="enginetype", y = "price" , data = carprice)

ax2.set(xlabel='Engine Type', ylabel='Average Car Price')

plt.show()
print(carprice['fuelsystem'].value_counts(),"\n")

print(carprice.groupby("fuelsystem").price.mean().sort_values(ascending=False))



plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)

ax1 = sns.countplot(x="fuelsystem", data = carprice)

ax1.set(xlabel='Fuel System', ylabel='Count of Cars')

plt.subplot(1,2,2)

ax2 = sns.barplot(x="fuelsystem", y = "price" , data = carprice)

ax2.set(xlabel='Fuel System', ylabel='Average Car Price')

plt.show()
# Derive a new column that is Fuel economy from citympg and highwaympg

carprice['fueleconomy'] = (0.55 * carprice['citympg']) + (0.45 * carprice['highwaympg'])



# Drop both citympg and highwaympg

carprice.drop(['citympg','highwaympg'], axis = 1, inplace = True)
def pp(x,y,z):    

    sns.pairplot(carprice, x_vars=[x,y,z], y_vars='price',size=4, aspect=1, kind='scatter')

    plt.show()



pp('carwidth', 'carlength', 'curbweight')

pp('carheight','enginesize', 'boreratio' )

pp('stroke','compressionratio', 'horsepower')

pp('peakrpm','wheelbase', 'fueleconomy')
plt.figure(figsize = (14, 8))

sns.heatmap(carprice.corr(), annot = True, cmap="YlGnBu")

plt.show()
# Create a new column called company category having values Budget, Mid_Range and Luxury based upon 

# company average price of their cars.

# If company average price < 10000 then Budget

# Else If company average price >= 10000 and < 20000 then Mid_Range

# Else If company average price > 20000 then Luxury

carprice["company_average_price"] = round(carprice.groupby('company')["price"].transform('mean'))



carprice['company_category'] = carprice["company_average_price"].apply(lambda x : "budget" if x < 10000

                                                                       else ("mid_range" if 10000 <= x < 20000

                                                                       else "luxury"))

plt.figure(figsize=(12, 6))

sns.boxplot(x = 'company_category', y = 'price', data = carprice)

plt.show()



print(carprice.groupby("company_category").company.count())
# Drop company and company_average_price after deriving company_category which will be used for modeling

carprice.drop(['company','company_average_price'], axis = 1, inplace = True)
# Based upon data visualization of drivewheel, derive a single numeric column of drivewheel_rwd where if 1 then it implies 

# rwd else 4wd or fwd

carprice["drivewheel_rwd"] = np.where(carprice["drivewheel"].str.contains("rwd"), 1, 0)



# Drop drivewheel column

carprice.drop(['drivewheel'], axis = 1, inplace = True)
# Based upon data visualization, derive a single numeric column of cylindernumber_four where if 1 then it implies 

# 4 cylinders car else not

carprice["cylindernumber_four"] = np.where(carprice["cylindernumber"].str.contains("four"), 1, 0)



# Drop cylindernumber column

carprice.drop(['cylindernumber'], axis = 1, inplace = True)
# Based upon data visualization, derive a single numeric column of enginetype_ohc where if 1 then it implies ohc

# cylinder type else not

carprice["enginetype_ohc"] = np.where(carprice["enginetype"].str.contains("ohc"), 1, 0)



# Drop enginetype column

carprice.drop(['enginetype'], axis = 1, inplace = True)
print(carprice.shape)

# Right now we have 23 columns

carprice.head(10)
# Create dummy variables for the remaining categorical variables

carprice_dummy = carprice.loc[:, ['company_category','doornumber','fueltype','aspiration','carbody','fuelsystem']]

carprice_dummy.head()

dummy = pd.get_dummies(carprice_dummy, drop_first = True)

print(dummy.shape)

dummy.head(10)
# Concatenate carprice and dummy data frames

carprice = pd.concat([carprice, dummy], axis = 1)



# Drop the original categorical columns once we have the corresponding derived numerical columns

carprice.drop(['company_category','doornumber','fueltype','aspiration','carbody','fuelsystem'], axis = 1, inplace = True)
carprice.head(10)
# Now we have 32 columns and all are numeric which can be used for modeling

carprice.shape
carprice.info()
# We specify this so that the train and test data set always have the same rows, respectively

np.random.seed(0)



df_train, df_test = train_test_split(carprice, train_size = 0.7, test_size = 0.3, random_state = 100)
scaler = MinMaxScaler()
# Apply scaler() to all the columns except the dummy variables and target variable

num_vars = ['wheelbase', 'carlength','carwidth','carheight','curbweight', 'enginesize', 'boreratio', 

            'stroke','compressionratio','horsepower','peakrpm','fueleconomy']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])



df_train.head()
# Set y_train to the target column

y_train = df_train.pop('price')

# Set X_train to the independent variables

X_train = df_train
# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 12

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 12) # running RFE

rfe = rfe.fit(X_train, y_train)
(list(zip(X_train.columns,rfe.support_,rfe.ranking_)))
col = X_train.columns[rfe.support_]

col
# Create function definitions to build model and check VIF

def build_model(X,y):

    X = sm.add_constant(X) #Adding the constant

    lm = sm.OLS(y,X).fit() # fitting the model

    print(lm.summary()) # model summary

    return X

    

def checkVIF(X):

    vif = pd.DataFrame()

    vif['Features'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    return(vif)
# Creating X_train_rfe dataframe with RFE selected variables

X_train_rfe = X_train[col]



#Build model and check VIF

X_train_lm = build_model(X_train_rfe,y_train)

checkVIF(X_train_lm)
X_train_lm = X_train_lm.drop(["fuelsystem_idi"], axis = 1)

print(X_train_lm.columns)



#Build model and check VIF

X_train_lm = build_model(X_train_lm,y_train)

checkVIF(X_train_lm)
X_train_lm = X_train_lm.drop(["carlength"], axis = 1)

print(X_train_lm.columns)



# Build model and check VIF

X_train_lm = build_model(X_train_lm,y_train)

checkVIF(X_train_lm)
X_train_lm = X_train_lm.drop(["fueleconomy"], axis = 1)

print(X_train_lm.columns)



# Build model and check VIF

X_train_lm = build_model(X_train_lm,y_train)

checkVIF(X_train_lm)
X_train_lm = X_train_lm.drop(["fueltype_gas"], axis = 1)

print(X_train_lm.columns)

X_train_lm = build_model(X_train_lm,y_train)

checkVIF(X_train_lm)
X_train_lm = X_train_lm.drop(["curbweight"], axis = 1)

print(X_train_lm.columns)

X_train_lm = build_model(X_train_lm,y_train)

checkVIF(X_train_lm)
X_train_lm = X_train_lm.drop(["peakrpm"], axis = 1)

print(X_train_lm.columns)

X_train_lm = build_model(X_train_lm,y_train)

checkVIF(X_train_lm)
X_train_lm = X_train_lm.drop(["carbody_sedan"], axis = 1)

print(X_train_lm.columns)

X_train_lm = build_model(X_train_lm,y_train)

checkVIF(X_train_lm)
X_train_lm = X_train_lm.drop(["carbody_wagon"], axis = 1)

print(X_train_lm.columns)

X_train_lm = build_model(X_train_lm,y_train)

checkVIF(X_train_lm)
lm = sm.OLS(y_train,X_train_lm).fit()



y_train_pred = lm.predict(X_train_lm)



residual = y_train_pred - y_train
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot(residual, bins = 20)

fig.suptitle('Train Data Distribution of Error Terms')                  # Plot heading 

plt.xlabel('Errors')       
# Plot the scatter plot of the error terms

fig = plt.figure()

sns.scatterplot(y_train, residual)

fig.suptitle('Train Data Scatter Plot of Error Terms')

plt.ylabel('Errors') 
num_vars = ['wheelbase', 'carlength','carwidth','carheight','curbweight', 'enginesize', 'boreratio', 

            'stroke','compressionratio','horsepower','peakrpm','fueleconomy']



df_test[num_vars] = scaler.transform(df_test[num_vars])
y_test = df_test.pop('price')

X_test = df_test
# Now let's use our model to make predictions.

X_train_lm = X_train_lm.drop(['const'], axis=1)



# Creating X_test_new dataframe by dropping variables from X_test using the final X_train_lm.columns

X_test_new = X_test[X_train_lm.columns]



# Adding a constant variable 

X_test_new = sm.add_constant(X_test_new)
# Making predictions

y_test_pred = lm.predict(X_test_new)

print(y_test_pred)
from sklearn.metrics import r2_score

r2_score(y_train, y_train_pred)
from sklearn.metrics import r2_score

r2_score(y_test, y_test_pred)
#Returns the mean squared error; we'll take a square root

print(np.sqrt(mean_squared_error(y_train, y_train_pred)))

np.sqrt(mean_squared_error(y_test, y_test_pred))
# Plotting y_test and y_test_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_train,y_train_pred)

fig.suptitle('y_train vs y_train_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_train', fontsize=18)                          # X-label

plt.ylabel('y_train_pred', fontsize=16)                     # Y-label



fig = plt.figure()

plt.scatter(y_test,y_test_pred)

fig.suptitle('y_test vs y_test_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_test_pred', fontsize=16)                     # Y-label
# Final Model Summary

lm.summary()
residual = y_test - y_test_pred

fig = plt.figure()

sns.distplot(residual, bins = 20)

fig.suptitle('Test Data Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)