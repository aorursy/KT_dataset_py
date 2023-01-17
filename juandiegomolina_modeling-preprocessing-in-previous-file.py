import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
data = pd.read_csv('../input/car-price-preprocessed/Car_price_preprocessed.csv')

data
pd.options.display.max_columns = None

pd.options.display.max_rows = None

data.describe()
sns.distplot(data['normalized-losses'])

plt.show()
q = data['normalized-losses'].quantile(0.99)

data = data[data['normalized-losses']<q]

sns.distplot(data['normalized-losses'])

plt.show()
sns.distplot(data['wheel-base'])

plt.show()
sns.distplot(data['length'])

plt.show()
sns.distplot(data['width'])

plt.show()
sns.distplot(data['height'])

plt.show()
sns.distplot(data['curb-weight'])

plt.show()
sns.distplot(data['engine-size'])

plt.show()
q = data['engine-size'].quantile(0.99)

data = data[data['engine-size']<q]

sns.distplot(data['engine-size'])

plt.show()
sns.distplot(data['bore'])

plt.show()
q = data['bore'].quantile(0.01)

data = data[data['bore']>q]

sns.distplot(data['bore'])

plt.show()
sns.distplot(data['stroke'])

plt.show()
sns.distplot(data['compression-ratio'])

plt.show()
data[data['compression-ratio'] >= 20]
data[data['compression-ratio'] <=13]
sns.distplot(data['horsepower'])

plt.show()
sns.distplot(data['peak-rpm'])

plt.show()
sns.distplot(data['city-mpg'])

plt.show()
sns.distplot(data['highway-mpg'])

plt.show()
data.columns.values
num_features = ['symboling', 'normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight',

       'num-of-cylinders', 'engine-size', 'bore', 'stroke', 'compression-ratio',

       'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg']
len(num_features)
for i in range(16):

    plt.scatter(data[num_features[i]], data['price'])

    plt.title('Price and {}'.format(num_features[i]))

    plt.show()
for i in range(16):

    plt.scatter(data[num_features[i]], np.log(data['price']))

    plt.title('Log Price and {}'.format(num_features[i]))

    plt.show()
for i in range(16):

    plt.scatter(np.log(data[num_features[i]]), np.log(data['price']))

    plt.title('Log Price and Log {}'.format(num_features[i]))

    plt.show()
data['Log-price'] = np.log(data['price'])

data.drop(['price'], axis=1, inplace=True)

data.head()
data.drop(['compression-ratio'], axis=1, inplace=True)

data.drop(['stroke'], axis=1, inplace=True)

data.drop(['peak-rpm'], axis=1, inplace=True)

data.head()
data['num-of-cylinders'].unique()
data['num-of-cylinders'] = data['num-of-cylinders'].map({4:0, 5:1, 6:1})

data.head()
data = data.rename(columns={'num-of-cylinders':'+4 cylinders'})

data.head()
data.drop(['city-mpg'], axis=1, inplace=True)

data['highway-mpg'] = np.log(data['highway-mpg'])

data.rename(columns={'highway-mpg':'Log highway-mpg'}, inplace=True)

data.head()
data.head()
plt.scatter(abs(data['symboling'] - 1) , data['Log-price'] ** 0.5)

plt.show()

data['symboling'] = abs(data['symboling'] - 1)

data.rename(columns={'symboling':'abs(symb-1)'}, inplace=True)

data.head()
linear_data = data.copy()
data.columns.values
from statsmodels.stats.outliers_influence import variance_inflation_factor



variables = data[['abs(symb-1)', 'normalized-losses',

       'wheel-base', 'length', 'width', 'height', 'curb-weight',

       'engine-size', 'bore', 'horsepower', 'Log highway-mpg']]



vif = pd.DataFrame()

vif['Features'] = variables.columns

vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif
variables = data[['abs(symb-1)','normalized-losses']]



vif = pd.DataFrame()

vif['Features'] = variables.columns

vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif
for i in ['wheel-base', 'length', 'width', 'height', 'curb-weight',

       'engine-size', 'bore', 'horsepower', 'Log highway-mpg']:

    data.drop([i], axis=1, inplace=True)

data.head()
targets = data['Log-price']

inputs = data.drop(['Log-price'], axis=1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(inputs)

scaled_inputs = scaler.transform(inputs)
scaled_inputs
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, test_size=0.2, random_state=97)
from sklearn.linear_model import LinearRegression

reg =  LinearRegression()

reg.fit(x_train, y_train)
y_hat = reg.predict(x_train)
# The simplest way to compare the targets (y_train) and the predictions (y_hat) is to plot them on a scatter plot

# The closer the points to the 45-degree line, the better the prediction

plt.scatter(y_train, y_hat)

# Let's also name the axes

plt.xlabel('Targets (y_train)',size=18)

plt.ylabel('Predictions (y_hat)',size=18)

plt.show()
# Another useful check of our model is a residual plot

# We can plot the PDF of the residuals and check for anomalies

sns.distplot(y_train - y_hat)



# Include a title

plt.title("Residuals PDF", size=18)
reg.score(x_train, y_train)
reg.intercept_
reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])

reg_summary['Weights'] = reg.coef_

reg_summary
y_hat_test = reg.predict(x_test)
plt.scatter(y_test, y_hat_test)

plt.xlabel('Targets (y_test)',size=18)

plt.ylabel('Predictions (y_hat_test)',size=18)

plt.xlim(8.5,10.5)

plt.ylim(8.5,10.5)

plt.show()
reg.score(x_test, y_test)
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])

df_pf.head()
y_test = y_test.reset_index(drop=True)

df_pf['Target'] = np.exp(y_test)
df_pf
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']

df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)

df_pf
df_pf.describe()