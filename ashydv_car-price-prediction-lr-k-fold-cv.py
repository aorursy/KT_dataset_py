# supress warnings

import warnings

warnings.filterwarnings('ignore')
# Importing all required packages

import numpy as np

import pandas as pd



# Data Visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from matplotlib.pyplot import xticks
df = pd.read_csv('../input/CarPrice_Assignment.csv')
df.head()
df.shape

# Data has 26 columns and 205 rows.
df.describe()
df.info()
df.columns
#checking duplicates

sum(df.duplicated(subset = 'car_ID')) == 0

# No duplicate values
# Checking Null values

df.isnull().sum()*100/df.shape[0]

# There are no NULL values in the dataset, hence it is clean.
df.price.describe()
sns.distplot(df['price'])
# Inference

# Mean and median of price are significantly different.

# Large standard deviation indicates that there is considerable variance in the prices of the automobiles.

# Price values are right-skewed, most cars are priced at the lower end (9000) of the price range.
# car_ID : Unique ID for each observation
# symboling : Its assigned insurance risk rating

#             A value of +3 indicates that the auto is risky,

#             -3 that it is probably pretty safe.(Categorical)
# Let's see the count of automobile in each category and percent share of each category.
fig, axs = plt.subplots(figsize = (10,5))

plt1 = sns.countplot(df['symboling'])

plt1.set(xlabel = 'Symbol', ylabel= 'Count of Cars')
df_sym = pd.DataFrame(df['symboling'].value_counts())

df_sym.plot.pie(subplots=True,labels = df_sym.index.values, autopct='%1.1f%%', figsize = (15,7.5))

# Unsquish the pie.

plt.gca().set_aspect('equal')

plt.show()

plt.tight_layout()
# Let's see average price of cars in each symbol category.
plt1 = df[['symboling','price']].groupby("symboling").mean().plot(kind='bar',legend = False,)

plt1.set_xlabel("Symbol")

plt1.set_ylabel("Avg Price (Dollars)")

xticks(rotation = 0)

plt.show()

# Inference

# More than 50% of cars are with symbol 0 or 1.

# Average price of car is lower for 0,1 & 2 symbol category.
df.CarName.values[0:10]
# It is observed that Car Name consists of two parts 'car company' + ' ' + 'Car Model'

# Let's split out car company to a new column.
df['brand'] = df.CarName.str.split(' ').str.get(0).str.upper()
len(set(df.brand.values))
# Let's see companies and their no of models.
fig, ax = plt.subplots(figsize = (15,5))

plt1 = sns.countplot(df['brand'], order=pd.value_counts(df['brand']).index,)

plt1.set(xlabel = 'Brand', ylabel= 'Count of Cars')

xticks(rotation = 90)

plt.show()

plt.tight_layout()
# It's noticed that in brand names,

# VOLKSWAGON has three different values as VOLKSWAGEN, VOKSWAGEN and VW

# MAZDA is also spelled as MAXDA

# PORSCHE as PORSCHE and PORCSCHE.

# Let's fix these data issues.

df['brand'] = df['brand'].replace(['VW', 'VOKSWAGEN'], 'VOLKSWAGEN')

df['brand'] = df['brand'].replace(['MAXDA'], 'MAZDA')

df['brand'] = df['brand'].replace(['PORCSHCE'], 'PORSCHE')

df['brand'] = df['brand'].replace(['TOYOUTA'], 'TOYOTA')
fig, ax = plt.subplots(figsize = (15,5))

plt1 = sns.countplot(df['brand'], order=pd.value_counts(df['brand']).index,)

plt1.set(xlabel = 'Brand', ylabel= 'Count of Cars')

xticks(rotation = 90)

plt.show()

plt.tight_layout()
df.brand.describe()
# Inference

# Toyota, a Japanese company has the most no of models.
# Let's see average car price of each company.


df_comp_avg_price = df[['brand','price']].groupby("brand", as_index = False).mean().rename(columns={'price':'brand_avg_price'})

plt1 = df_comp_avg_price.plot(x = 'brand', kind='bar',legend = False, sort_columns = True, figsize = (15,3))

plt1.set_xlabel("Brand")

plt1.set_ylabel("Avg Price (Dollars)")

xticks(rotation = 90)

plt.show()
#df_comp_avg_price
df = df.merge(df_comp_avg_price, on = 'brand')
df['brand_category'] = df['brand_avg_price'].apply(lambda x : "Budget" if x < 10000 

                                                     else ("Mid_Range" if 10000 <= x < 20000

                                                           else "Luxury"))

# Inference:

# Toyota has considerably high no of models in the market.

# Brands can be categorised as Luxury, Mid Ranged, Budget based on their average price.

# Some of the Luxury brans are
# Let's see how price varies with  Fuel Type
df_fuel_avg_price = df[['fueltype','price']].groupby("fueltype", as_index = False).mean().rename(columns={'price':'fuel_avg_price'})

plt1 = df_fuel_avg_price.plot(x = 'fueltype', kind='bar',legend = False, sort_columns = True)

plt1.set_xlabel("Fuel Type")

plt1.set_ylabel("Avg Price (Dollars)")

xticks(rotation = 0)

plt.show()
# Inference

# Diesel cars are priced more than gas cars.
df_aspir_avg_price = df[['aspiration','price']].groupby("aspiration", as_index = False).mean().rename(columns={'price':'aspir_avg_price'})

plt1 = df_aspir_avg_price.plot(x = 'aspiration', kind='bar',legend = False, sort_columns = True)

plt1.set_xlabel("Aspiration")

plt1.set_ylabel("Avg Price (Dollars)")

xticks(rotation = 0)

plt.show()
# Inference

# Cars with turbo aspiration engine are priced more than standard ones.
df_door_avg_price = df[['doornumber','price']].groupby("doornumber", as_index = False).mean().rename(columns={'price':'door_avg_price'})

plt1 = df_door_avg_price.plot(x = 'doornumber', kind='bar',legend = False, sort_columns = True)

plt1.set_xlabel("No of Doors")

plt1.set_ylabel("Avg Price (Dollars)")

xticks(rotation = 0)

plt.show()
# Inference

# Number of doors doesn't seem to have much effect on price.
df_body_avg_price = df[['carbody','price']].groupby("carbody", as_index = False).mean().rename(columns={'price':'carbody_avg_price'})

plt1 = df_body_avg_price.plot(x = 'carbody', kind='bar',legend = False, sort_columns = True)

plt1.set_xlabel("Car Body")

plt1.set_ylabel("Avg Price (Dollars)")

xticks(rotation = 0)

plt.show() 
# Inference 

# Hardtop and convertible are the most expensive whereas hatchbacks are the cheapest.
df_drivewheel_avg_price = df[['drivewheel','price']].groupby("drivewheel", as_index = False).mean().rename(columns={'price':'drivewheel_avg_price'})

plt1 = df_drivewheel_avg_price.plot(x = 'drivewheel', kind='bar', sort_columns = True,legend = False,)

plt1.set_xlabel("Drive Wheel Type")

plt1.set_ylabel("Avg Price (Dollars)")

xticks(rotation = 0)

plt.show()
# Inference

# Cars with Rear wheel drive have a higher price value.
plt1 = sns.scatterplot(x = 'wheelbase', y = 'price', data = df)

plt1.set_xlabel('Wheelbase (Inches)')

plt1.set_ylabel('Price of Car (Dollars)')

plt.show()
# Most cars has a wheel base around 95 inches.

# Price has a slight positive correlation with wheelbase.
# Let's see how price varies with Car's length, width,height and weight.
fig, axs = plt.subplots(2,2,figsize=(15,10))

plt1 = sns.scatterplot(x = 'carlength', y = 'price', data = df, ax = axs[0,0])

plt1.set_xlabel('Length of Car (Inches)')

plt1.set_ylabel('Price of Car (Dollars)')

plt2 = sns.scatterplot(x = 'carwidth', y = 'price', data = df, ax = axs[0,1])

plt2.set_xlabel('Width of Car (Inches)')

plt2.set_ylabel('Price of Car (Dollars)')

plt3 = sns.scatterplot(x = 'carheight', y = 'price', data = df, ax = axs[1,0])

plt3.set_xlabel('Height of Car (Inches)')

plt3.set_ylabel('Price of Car (Dollars)')

plt3 = sns.scatterplot(x = 'curbweight', y = 'price', data = df, ax = axs[1,1])

plt3.set_xlabel('Weight of Car (Pounds)')

plt3.set_ylabel('Price of Car (Dollars)')

plt.tight_layout()
# Inference

# Length width and weight of the car is positively related with the price.

# There is not much of a correlation with Height of the car with price.
fig, axs = plt.subplots(1,3,figsize=(20,5))

#

df_engine_avg_price = df[['enginetype','price']].groupby("enginetype", as_index = False).mean().rename(columns={'price':'engine_avg_price'})

plt1 = df_engine_avg_price.plot(x = 'enginetype', kind='bar', sort_columns = True, legend = False, ax = axs[0])

plt1.set_xlabel("Engine Type")

plt1.set_ylabel("Avg Price (Dollars)")

xticks(rotation = 0)

#

df_cylindernumber_avg_price = df[['cylindernumber','price']].groupby("cylindernumber", as_index = False).mean().rename(columns={'price':'cylindernumber_avg_price'})

plt1 = df_cylindernumber_avg_price.plot(x = 'cylindernumber', kind='bar', sort_columns = True,legend = False, ax = axs[1])

plt1.set_xlabel("Cylinder Number")

plt1.set_ylabel("Avg Price (Dollars)")

xticks(rotation = 0)

#

df_fuelsystem_avg_price = df[['fuelsystem','price']].groupby("fuelsystem", as_index = False).mean().rename(columns={'price':'fuelsystem_avg_price'})

plt1 = df_fuelsystem_avg_price.plot(x = 'fuelsystem', kind='bar', sort_columns = True,legend = False, ax = axs[2])

plt1.set_xlabel("Fuel System")

plt1.set_ylabel("Avg Price (Dollars)")

xticks(rotation = 0)

plt.show()
# Inference

# DOHCV and OHCV engine types are priced high.

# Eight and twelve cylinder cars have higher price.

# IDI and MPFI fuel system have higher price.

fig, axs = plt.subplots(3,2,figsize=(20,20))

#

plt1 = sns.scatterplot(x = 'enginesize', y = 'price', data = df, ax = axs[0,0])

plt1.set_xlabel('Size of Engine (Cubic Inches)')

plt1.set_ylabel('Price of Car (Dollars)')

#

plt2 = sns.scatterplot(x = 'boreratio', y = 'price', data = df, ax = axs[0,1])

plt2.set_xlabel('Bore Ratio')

plt2.set_ylabel('Price of Car (Dollars)')

#

plt3 = sns.scatterplot(x = 'stroke', y = 'price', data = df, ax = axs[1,0])

plt3.set_xlabel('Stroke')

plt3.set_ylabel('Price of Car (Dollars)')

#

plt4 = sns.scatterplot(x = 'compressionratio', y = 'price', data = df, ax = axs[1,1])

plt4.set_xlabel('Compression Ratio')

plt4.set_ylabel('Price of Car (Dollars)')

#

plt5 = sns.scatterplot(x = 'horsepower', y = 'price', data = df, ax = axs[2,0])

plt5.set_xlabel('Horsepower')

plt5.set_ylabel('Price of Car (Dollars)')

#

plt5 = sns.scatterplot(x = 'peakrpm', y = 'price', data = df, ax = axs[2,1])

plt5.set_xlabel('Peak RPM')

plt5.set_ylabel('Price of Car (Dollars)')

plt.tight_layout()

plt.show()

# Inference

# Size of Engine, bore ratio, and Horsepower has positive correlation with price.
# A single variable mileage can be calculated taking the weighted average of 55% city and 45% highways.
df['mileage'] = df['citympg']*0.55 + df['highwaympg']*0.45
# Let's see how price varies with mileage.
plt1 = sns.scatterplot(x = 'mileage', y = 'price', data = df)

plt1.set_xlabel('Mileage')

plt1.set_ylabel('Price of Car (Dollars)')

plt.show()
# Inference 

# Mileage has a negative correlation with price.
# It is expected that luxury brands don't care about mileage. Let's find out how price varies with brand category and mileage.
plt1 = sns.scatterplot(x = 'mileage', y = 'price', hue = 'brand_category', data = df)

plt1.set_xlabel('Mileage')

plt1.set_ylabel('Price of Car (Dollars)')

plt.show()
plt1 = sns.scatterplot(x = 'horsepower', y = 'price', hue = 'brand_category', data = df)

plt1.set_xlabel('Horsepower')

plt1.set_ylabel('Price of Car (Dollars)')

plt.show()
plt1 = sns.scatterplot(x = 'mileage', y = 'price', hue = 'fueltype', data = df)

plt1.set_xlabel('Mileage')

plt1.set_ylabel('Price of Car (Dollars)')

plt.show()
plt1 = sns.scatterplot(x = 'horsepower', y = 'price', hue = 'fueltype', data = df)

plt1.set_xlabel('Horsepower')

plt1.set_ylabel('Price of Car (Dollars)')

plt.show()
auto = df[['fueltype', 'aspiration', 'carbody', 'drivewheel', 'wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginetype',

       'cylindernumber', 'enginesize',  'boreratio', 'horsepower', 'price', 'brand_category', 'mileage']]
auto.head()
plt.figure(figsize=(15, 15))

sns.pairplot(auto)

plt.show()
plt.figure(figsize=(10, 20))

plt.subplot(4,2,1)

sns.boxplot(x = 'fueltype', y = 'price', data = auto)

plt.subplot(4,2,2)

sns.boxplot(x = 'aspiration', y = 'price', data = auto)

plt.subplot(4,2,3)

sns.boxplot(x = 'carbody', y = 'price', data = auto)

plt.subplot(4,2,4)

sns.boxplot(x = 'drivewheel', y = 'price', data = auto)

plt.subplot(4,2,5)

sns.boxplot(x = 'enginetype', y = 'price', data = auto)

plt.subplot(4,2,6)

sns.boxplot(x = 'brand_category', y = 'price', data = auto)

plt.subplot(4,2,7)

sns.boxplot(x = 'cylindernumber', y = 'price', data = auto)

plt.tight_layout()

plt.show()

# Categorical Variables are converted into Neumerical Variables with the help of Dummy Variable 
cyl_no = pd.get_dummies(auto['cylindernumber'], drop_first = True)
auto = pd.concat([auto, cyl_no], axis = 1)
brand_cat = pd.get_dummies(auto['brand_category'], drop_first = True)
auto = pd.concat([auto, brand_cat], axis = 1)
eng_typ = pd.get_dummies(auto['enginetype'], drop_first = True)
auto = pd.concat([auto, eng_typ], axis = 1)
drwh = pd.get_dummies(auto['drivewheel'], drop_first = True)
auto = pd.concat([auto, drwh], axis = 1)
carb = pd.get_dummies(auto['carbody'], drop_first = True)
auto = pd.concat([auto, carb], axis = 1)
asp = pd.get_dummies(auto['aspiration'], drop_first = True)
auto = pd.concat([auto, asp], axis = 1)
fuelt = pd.get_dummies(auto['fueltype'], drop_first = True)
auto = pd.concat([auto, fuelt], axis = 1)
auto.drop(['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginetype', 'cylindernumber','brand_category'], axis = 1, inplace = True)
from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively

np.random.seed(0)

df_train, df_test = train_test_split(auto, train_size = 0.7, test_size = 0.3, random_state = 100)
# We will use min-max scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Apply scaler() to all the columns except the 'dummy' variables

num_vars = ['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize','boreratio', 'horsepower', 'price','mileage']



df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.head()
# Let's check the correlation coefficients to see which variables are highly correlated



plt.figure(figsize = (16, 10))

sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")

plt.show()
y_train = df_train.pop('price')

X_train = df_train
from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import scale

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline



import warnings # supress warnings

warnings.filterwarnings('ignore')
# number of features

len(X_train.columns)
# creating a KFold object with 5 splits 

folds = KFold(n_splits = 5, shuffle = True, random_state = 100)



# specify range of hyperparameters

hyper_params = [{'n_features_to_select': list(range(2, 30))}]



# specify model

lm = LinearRegression()

lm.fit(X_train, y_train)

rfe = RFE(lm)             



# set up GridSearchCV()

model_cv = GridSearchCV(estimator = rfe, 

                        param_grid = hyper_params, 

                        scoring= 'r2', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True)      



# fit the model

model_cv.fit(X_train, y_train)                  
# cv results

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# plotting cv results

plt.figure(figsize=(16,6))



plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])

plt.xlabel('number of features')

plt.ylabel('r-squared')

plt.title("Optimal Number of Features")

plt.legend(['test score', 'train score'], loc='upper left')
# According to the above plot optimum number of features are 5.
# Running RFE with the output number of the variable equal to 10

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 5)             # running RFE

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]

col
# Creating X_test dataframe with RFE selected variables

X_train_rfe = X_train[col]
# Adding a constant variable 

import statsmodels.api as sm  

X_train_rfe = sm.add_constant(X_train_rfe)
lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model
#Let's see the summary of our linear model

print(lm.summary())
# Calculate the VIFs for the new model

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_rfe

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_price = lm.predict(X_train_rfe)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)                         # X-label
# Error terms are centered around 0, and are normally distributed. Hence Residual Analysis suggests model is fine.
num_vars = ['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize','boreratio', 'horsepower', 'price','mileage']



df_test[num_vars] = scaler.transform(df_test[num_vars])
y_test = df_test.pop('price')

X_test = df_test
# Now let's use our model to make predictions.



# Creating X_test_new dataframe by dropping variables from X_test

X_test_new = X_test[['curbweight', 'carwidth', 'horsepower', 'Luxury','dohcv']]



# Adding a constant variable 

X_test_new = sm.add_constant(X_test_new)
# Making predictions

y_pred = lm.predict(X_test_new)
from sklearn.metrics import r2_score 

r2_score(y_test, y_pred)
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)                          # Y-label