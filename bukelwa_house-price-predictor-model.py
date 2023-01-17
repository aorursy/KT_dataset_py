#first let us import all the packages that we need 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split

file = pd.read_csv('../input/train.csv')
# We are going to look at the first 5 rows of te dataset to see how catagorical and variant columns we have 
file.head()
# to have a better view of the type of each column we will use info, this will also help us see if we have missing values in columns
file.info()
# we also need to check the names of all the columns in the dataset
file.columns
# we can use decribe to see the MIn, Max, Mean of each continuous column.
# now we have an idea of which columns are continuous
continuous = file.describe()
continuous
#Here we are going to identify all the columns with NaN values 
missing = file.columns[file.isna().any()].tolist()
missing
#let us take a closer look at these columns to figure out if our data is missing at random 

missing_values = file[['LotFrontage',
 'Alley',
 'MasVnrType',
 'MasVnrArea',
 'BsmtQual',
 'BsmtCond',
 'BsmtExposure',
 'BsmtFinType1',
 'BsmtFinType2',
 'Electrical',
 'FireplaceQu',
 'GarageType',
 'GarageYrBlt',
 'GarageFinish',
 'GarageQual',
 'GarageCond',
 'PoolQC',
 'Fence',
 'MiscFeature']]
missing_values.tail()
#here we had a look at all numeric columns
file_numeric = file.select_dtypes(include=[np.number])
file_numeric.head()
file_numeric.columns
missing_numeric = file_numeric.columns[file_numeric.isna().any()].tolist()
missing_numeric
file_numeric[['LotFrontage', 'MasVnrArea', 'GarageYrBlt']].head()
#correlation matrix
corrmat = file_numeric.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
# We are going to plot apair plot to see the Histograms and scatterplot of each variable. 
# This will also help us understand which variables are correlated 

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'YearBuilt', 'LotArea']
sns.pairplot(file_numeric[cols], size = 2.5)
plt.show();
#here we are transforming the sale price distribution
file_numeric['SalePrice'] = np.log(file_numeric['SalePrice'])
tobe_normalised = file_numeric[[ 'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'YearBuilt', 'LotArea']]

# here we want to normalise the variables that we are going to use in the Multi Regression 

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(tobe_normalised)
df_normalized = pd.DataFrame(np_scaled)

df_normalized.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
df_normalized.head()
#here we will run a multy regression on our variables 

x = np.column_stack((df_normalized['x1'],df_normalized['x2'],df_normalized['x3'],df_normalized['x4'],df_normalized['x5'],df_normalized['x6']))

x = sm.add_constant(x, prepend=True)

y = file_numeric['SalePrice']
results = smf.OLS(y,x).fit()
print(results.summary())
regressor = DecisionTreeRegressor()
regressor=regressor.fit(df_normalized,y)

mean_squared_error(regressor.predict(df_normalized), y)
x_train , x_test , y_train, y_test = train_test_split(df_normalized, y, test_size=0.2, random_state=7)
regressor = DecisionTreeRegressor()
regressor=regressor.fit(x_train, y_train)

training_error = mean_squared_error(regressor.predict(x_train), y_train)
test_set_error = mean_squared_error(regressor.predict(x_test), y_test)
print(training_error,test_set_error)
y_train.mean()
