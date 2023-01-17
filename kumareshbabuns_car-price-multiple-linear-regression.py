# Import Packages

import pandas as pd              # Data Analysis package

import numpy as np               

import matplotlib.pyplot as plt  # Data Virtualization package

%matplotlib inline 

import seaborn as sns            # Data Virtualization package

import warnings                  # Supress Warnings

warnings.filterwarnings('ignore')
# Create dataframe by reading the car price dataset

car_details = pd.read_csv('../input/CarPrice_Assignment.csv')
# Read the first 5 observations from the dataframe

car_details.head()
# Print the shape of the car_details dataframe

car_details.shape
# Describe car_details dataframe

car_details.describe()
# Get the detailed information

car_details.info()
# Split the company name from CarName variable

companyname = car_details['CarName'].apply(lambda name : name.split(' ')[0])
# Dropping the CarName variable as it is not needed

car_details.drop(columns = {'CarName'}, axis = 1, inplace = True)
# Adding the companyname as a new variable

car_details.insert(loc = 3, column = 'companyname', value = companyname)
# Get the list of first 5 observations

car_details.head()
# Check the unique values in companyname variable

car_details['companyname'].unique()
# Convert the data into lowercase

car_details['companyname'] = car_details['companyname'].str.lower()
# Define a function to rename the spelling mistakes

def renameCompanyName(error_data, correct_data):

  car_details['companyname'].replace(error_data, correct_data, inplace = True)
# Call renameCompanyName function

renameCompanyName('vw','volkswagen')

renameCompanyName('vokswagen','volkswagen')

renameCompanyName('maxda','mazda')

renameCompanyName('porcshce','porsche')

renameCompanyName('toyouta','toyota')
# Check the unique values in companyname variable

car_details['companyname'].unique()
# Checking for duplicate values in car_details dataframe

car_details.loc[car_details.duplicated()]
# Let's understand the price of the car

plt.figure(figsize=(15,6)) # Set width and height for the plots



plt.subplot(1,2,1) # Set the rows, columns and their indexing position

sns.distplot(a = car_details.price)



plt.subplot(1,2,2) # Set the rows, columns and their indexing position

sns.boxplot(y = car_details.price)
# Let's see the mean, median and other percentile for the car prices

car_details.price.describe(percentiles = [0.25, 0.5, 0.75, 0.85, 0.95, 1])
# Let's virtualize the car companies, car types and fuel types

plt.figure(figsize = (20,6))



plt.subplot(1,3,1)

plt1 = car_details.companyname.value_counts().plot('bar')

plt.title('Companies')

plt1.set(xlabel = 'Car Company', ylabel='Frequency of Car Company')



plt.subplot(1,3,2)

plt1 = car_details.carbody.value_counts().plot('bar')

plt.title('Car Type')

plt1.set(xlabel = 'Car Type', ylabel='Frequency of Car type')



plt.subplot(1,3,3)

plt1 = car_details.fueltype.value_counts().plot('bar')

plt.title('Fuel Type')

plt1.set(xlabel = 'Fuel Type', ylabel='Frequency of fuel type')
# Let's virutalize the engine types

plt1 = car_details.enginetype.value_counts().plot('bar')

plt.title('Engine Type')

plt1.set(xlabel = 'Engine Type', ylabel='Frequency of Engine Type')
plt.figure(figsize=(20,6))



plt.subplot(1,3,1)

plt1 = car_details.groupby('companyname')['price'].mean().sort_values(ascending = False).plot('bar')

plt1.set(xlabel = 'Car Company', ylabel = 'Average Price')



plt.subplot(1,3,2)

plt1 = car_details.groupby('enginetype')['price'].mean().sort_values(ascending = False).plot('bar')

plt1.set(xlabel = 'Engine Type', ylabel = 'Average Price')



plt.subplot(1,3,3)

plt1 = car_details.groupby('fueltype')['price'].mean().sort_values(ascending = False).plot('bar')

plt1.set(xlabel = 'Fuel Type', ylabel = 'Average Price')
plt.figure(figsize=(18,5))



plt.subplot(1,2,1)

plt1 = car_details.enginelocation.value_counts().sort_values(ascending = False).plot('bar')

plt1.set(xlabel = 'Engine Location', ylabel = 'Frequency of Engine Location')



plt.subplot(1,2,2)

plt1 = car_details.groupby('enginelocation')['price'].mean().sort_values(ascending = False).plot('bar')

plt1.set(xlabel = 'Engine Location', ylabel = 'Average Price')
# Calculating the fuel economy by using highwaympg and citympg

car_details['fueleconomy'] = (0.45 * car_details['highwaympg']) + (0.55 * car_details['citympg'])
# Calculating the stroke ratio by using boreratio and stroke

car_details['strokeratio'] = car_details['boreratio'] / car_details['stroke']
# Categorizing the car companies based on average car price

car_details['price'] = car_details['price'].astype('int')

temp1 = car_details.copy()

temp2 = temp1.groupby('companyname')['price'].mean()

temp1 = temp1.merge(temp2.reset_index(), how = 'left', on = 'companyname')

bins = [0, 10000, 20000, 40000]

cars_bins = ['Low', 'Medium', 'High']

car_details['carsrange'] = pd.cut(temp1['price_y'], bins, right = False, labels = cars_bins)
plt.figure(figsize = (15,6))

plt.title('Fuel Economy vs Price')

sns.scatterplot(x = car_details['fueleconomy'], y = car_details['price'])

plt.xlabel('Fuel Economy')

plt.ylabel('Price')
plt.figure(figsize=(15,6))

sns.heatmap(car_details.corr(), annot = True, cmap='YlGnBu')

categorical_variables = ['symboling', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation',

                       'enginetype', 'cylindernumber', 'fuelsystem', 'carsrange']



def dummies(x,df):

    temp = pd.get_dummies(df[x], drop_first = True)

    df= pd.concat([df, temp], axis = 1)

    df.drop([x], axis = 1, inplace = True)

    return df

  

for variable in categorical_variables:

   car_details = dummies(variable, car_details)

car_details.shape
car_details.head()
#Removing car_ID and companyname as it is not required for model building

car_details.drop(columns =['car_ID','companyname'], inplace = True)
car_details.shape
# Importing train_test_split to train the data for model building

from sklearn.model_selection import train_test_split



np.random.seed(0)

df_train, df_test = train_test_split(car_details, train_size = 0.7, test_size = 0.3, random_state = 100)
# Use MinMaxScaler to apply scaling

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

num_vars = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio',

            'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'fueleconomy', 'strokeratio', 'price']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.describe()
df_train.head()
#Dividing data into X and y variables

y_train = df_train.pop('price')

X_train = df_train
#RFE

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm 

from statsmodels.stats.outliers_influence import variance_inflation_factor
lm = LinearRegression()

lm.fit(X_train,y_train)

rfe = RFE(lm, 10)

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
X_train.columns[rfe.support_]
X_train_rfe = X_train[X_train.columns[rfe.support_]]

X_train_rfe.head()
def buildModel(X,y):

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
X_train_new = buildModel(X_train_rfe,y_train)
X_train_new = X_train_new.drop(['hardtop'], axis = 1)
X_train_new = buildModel(X_train_new,y_train)
vif_df = X_train_new.drop(['const'], axis = 1)
checkVIF(vif_df)
X_train_new = X_train_new.drop(['curbweight'], axis = 1)
X_train_new = buildModel(X_train_new,y_train)
X_train_new = X_train_new.drop(['wagon'], axis = 1)
X_train_new = buildModel(X_train_new,y_train)
vif_df = X_train_new.drop(['const'], axis = 1)
checkVIF(vif_df)
X_train_new = X_train_new.drop(['horsepower'], axis = 1)
X_train_new = buildModel(X_train_new,y_train)
X_train_new = X_train_new.drop(['hatchback'], axis = 1)
X_train_new = buildModel(X_train_new,y_train)
X_train_new = X_train_new.drop(['three'], axis = 1)
X_train_new = buildModel(X_train_new,y_train)
X_train_new = X_train_new.drop(['dohcv'], axis = 1)
X_train_new = buildModel(X_train_new,y_train)
vif_df = X_train_new.drop(['const'], axis = 1)
checkVIF(vif_df)
lm = sm.OLS(y_train,X_train_new).fit()

y_train_price = lm.predict(X_train_new)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  

plt.xlabel('Errors', fontsize = 18) 
# Scaling the test data

num_vars = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio',

            'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'fueleconomy', 'strokeratio', 'price']

df_test[num_vars] = scaler.fit_transform(df_test[num_vars])
#Dividing into X and y

y_test = df_test.pop('price')

X_test = df_test
# Now let's use our model to make predictions.

X_train_new = X_train_new.drop('const',axis=1)

# Creating X_test_new dataframe by dropping variables from X_test

X_test_new = X_test[X_train_new.columns]



# Adding a constant variable 

X_test_new = sm.add_constant(X_test_new)
# Making predictions

y_pred = lm.predict(X_test_new)
from sklearn.metrics import r2_score 

r2_score(y_test, y_pred)
print(lm.summary())
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)              

plt.xlabel('y_test', fontsize=18)                          

plt.ylabel('y_pred', fontsize=16)  