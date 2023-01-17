# Import libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import sklearn 

import matplotlib.pyplot as plt

%matplotlib inline
# Read data

dataset = pd.read_csv('../input/Melbourne_housing_extra_data-18-08-2017.csv')
# Number of rows and columns

print(dataset.shape)



# View first few records

dataset.head()
# View data types

dataset.info()
# Identify object columns

print(dataset.select_dtypes(['object']).columns)
# Convert objects to categorical variables

obj_cats = ['Suburb', 'Address', 'Type', 'Method', 'SellerG', 'CouncilArea','Regionname']



for colname in obj_cats:

    dataset[colname] = dataset[colname].astype('category')  
# Convert to date object

dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset.describe().transpose()
# Convert numeric variables to categorical

num_cats = ['Postcode']  



for colname in num_cats:

    dataset[colname] = dataset[colname].astype('category')   



# Confirm changes

dataset.info()
# Examine Rooms v Bedroom2

dataset['Rooms v Bedroom2'] = dataset['Rooms'] - dataset['Bedroom2']

dataset
# Drop columns

dataset = dataset.drop(['Bedroom2','Rooms v Bedroom2'],1)
# Add age variable

dataset['Age'] = 2017 - dataset['YearBuilt']



# Identify historic homes

dataset['Historic'] = np.where(dataset['Age']>=50,'Historic','Contemporary')



# Convert to Category

dataset['Historic'] = dataset['Historic'].astype('category')
# Number of entries

dataset.info()
# Visualize missing values

fig, ax = plt.subplots(figsize=(15,7))

sns.set(font_scale=1.2)

sns.heatmap(dataset.isnull(),yticklabels = False, cbar = False, cmap = 'Greys_r')

plt.show()
# Count of missing values

dataset.isnull().sum()
# Percentage of missing values

dataset.isnull().sum()/len(dataset)*100
# View missing data

#dataset[dataset['Bedroom2'].isnull()]

#To remove rows missing data in a specific column 

# dataset =dataset[pd.notnull(dataset['Price'])]



# To remove an entire column

#dataset = dataset.drop('Bedroom2',axis = 1)



# Remove rows missing data

dataset = dataset.dropna()



# Confirm that observations missing data were removed  

dataset.info()
dataset.describe().transpose()
dataset[dataset['Age']>800]
dataset[dataset['BuildingArea']==0]
dataset[dataset['Landsize']==0]
# Remove outlier

dataset = dataset[dataset['BuildingArea']!=0]



# Confirm removal

dataset.describe().transpose()
plt.figure(figsize=(16,7))

sns.distplot(dataset['Price'], kde = False,hist_kws=dict(edgecolor="k"))
# Identify categorical features

dataset.select_dtypes(['category']).columns
# Abbreviate Regionname categories

dataset['Regionname'] = dataset['Regionname'].map({'Northern Metropolitan':'N Metro',

                                            'Western Metropolitan':'W Metro', 

                                            'Southern Metropolitan':'S Metro', 

                                            'Eastern Metropolitan':'E Metro', 

                                            'South-Eastern Metropolitan':'SE Metro', 

                                            'Northern Victoria':'N Vic',

                                            'Eastern Victoria':'E Vic',

                                            'Western Victoria':'W Vic'})
# Suplots of categorical features v price

sns.set_style('darkgrid')

f, axes = plt.subplots(2,2, figsize = (15,15))



# Plot [0,0]

sns.boxplot(data = dataset, x = 'Type', y = 'Price', ax = axes[0,0])

axes[0,0].set_xlabel('Type')

axes[0,0].set_ylabel('Price')

axes[0,0].set_title('Type v Price')



# Plot [0,1]

sns.boxplot(x = 'Method', y = 'Price', data = dataset, ax = axes[0,1])

axes[0,1].set_xlabel('Method')

#axes[0,1].set_ylabel('Price')

axes[0,1].set_title('Method v Price')



# Plot [1,0]

sns.boxplot(x = 'Regionname', y = 'Price', data = dataset, ax = axes[1,0])

axes[1,0].set_xlabel('Regionname')

#axes[1,0].set_ylabel('Price')

axes[1,0].set_title('Region Name v Price')



# Plot [1,1]

sns.boxplot(x = 'Historic', y = 'Price', data = dataset, ax = axes[1,1])

axes[1,1].set_xlabel('Historic')

axes[1,1].set_ylabel('Price')

axes[1,1].set_title('Historic v Price')



plt.show()
# Identify numeric features

dataset.select_dtypes(['float64','int64']).columns
# Suplots of numeric features v price

sns.set_style('darkgrid')

f, axes = plt.subplots(4,2, figsize = (20,30))



# Plot [0,0]

axes[0,0].scatter(x = 'Rooms', y = 'Price', data = dataset, edgecolor = 'b')

axes[0,0].set_xlabel('Rooms')

axes[0,0].set_ylabel('Price')

axes[0,0].set_title('Rooms v Price')



# Plot [0,1]

axes[0,1].scatter(x = 'Distance', y = 'Price', data = dataset, edgecolor = 'b')

axes[0,1].set_xlabel('Distance')

# axes[0,1].set_ylabel('Price')

axes[0,1].set_title('Distance v Price')



# Plot [1,0]

axes[1,0].scatter(x = 'Bathroom', y = 'Price', data = dataset, edgecolor = 'b')

axes[1,0].set_xlabel('Bathroom')

axes[1,0].set_ylabel('Price')

axes[1,0].set_title('Bathroom v Price')



# Plot [1,1]

axes[1,1].scatter(x = 'Car', y = 'Price', data = dataset, edgecolor = 'b')

axes[1,0].set_xlabel('Car')

axes[1,1].set_ylabel('Price')

axes[1,1].set_title('Car v Price')



# Plot [2,0]

axes[2,0].scatter(x = 'Landsize', y = 'Price', data = dataset, edgecolor = 'b')

axes[2,0].set_xlabel('Landsize')

axes[2,0].set_ylabel('Price')

axes[2,0].set_title('Landsize v  Price')



# Plot [2,1]

axes[2,1].scatter(x = 'BuildingArea', y = 'Price', data = dataset, edgecolor = 'b')

axes[2,1].set_xlabel('BuildingArea')

axes[2,1].set_ylabel('BuildingArea')

axes[2,1].set_title('BuildingArea v Price')



# Plot [3,0]

axes[3,0].scatter(x = 'Age', y = 'Price', data = dataset, edgecolor = 'b')

axes[3,0].set_xlabel('Age')

axes[3,0].set_ylabel('Price')

axes[3,0].set_ylabel('Age v Price')



# Plot [3,1]

axes[3,1].scatter(x = 'Propertycount', y = 'Price', data = dataset, edgecolor = 'b')

axes[3,1].set_xlabel('Propertycount')

#axes[3,1].set_ylabel('Price')

axes[3,1].set_title('Property Count v Price')



plt.show()
# Pairplot

#sns.pairplot(dataset,vars= ['Rooms', 'Price', 'Distance', 'Bathroom', 'Car', 'Landsize','BuildingArea',  'Propertycount','Age'], palette = 'viridis')
plt.figure(figsize=(10,6))

sns.heatmap(dataset.corr(),cmap = 'coolwarm',linewidth = 1,annot= True, annot_kws={"size": 9})

plt.title('Variable Correlation')
# Identify numeric features

dataset.select_dtypes(['float64','int64']).columns
# Split

# Create features variable 

X =dataset[['Rooms', 'Distance', 'Bathroom', 'Car', 'Landsize', 

            'BuildingArea', 'Propertycount','Age']]



# Create target variable

y = dataset['Price']



# Train, test, split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .20, random_state= 0)
# Fit

# Import model

from sklearn.linear_model import LinearRegression



# Create linear regression object

regressor = LinearRegression()



# Fit model to training data

regressor.fit(X_train,y_train)
# Predict

# Predicting test set results

y_pred = regressor.predict(X_test)
# Score It

from sklearn import metrics

print('MAE:',metrics.mean_absolute_error(y_test,y_pred))

print('MSE:',metrics.mean_squared_error(y_test,y_pred))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
# Calculated R Squared

print('R^2 =',metrics.explained_variance_score(y_test,y_pred))
# Actual v predictions scatter

plt.scatter(y_test, y_pred)
# Histogram of the distribution of residuals

sns.distplot((y_test - y_pred))
cdf = pd.DataFrame(data = regressor.coef_, index = X.columns, columns = ['Coefficients'])

cdf