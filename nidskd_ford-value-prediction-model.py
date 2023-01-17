# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Importing clean Ford data set. Duplicates and bad entries have already been removed.

filepath = r'/kaggle/input/used-car-dataset-ford-and-mercedes/ford.csv'
ford = pd.read_csv(filepath)
ford
# an indication of the cars in the dataset
print(ford['model'].value_counts())
# another broad oulook over data set.
ford.describe()
# scaling before category binarisation
import sklearn
from sklearn.preprocessing import MinMaxScaler
dfTest = ford.copy()
dfTest[['year', 'price','mileage','tax','mpg','engineSize']] = scaler.fit_transform(dfTest[['year', 'price','mileage','tax','mpg','engineSize']])
dfTest
import category_encoders as ce
ford_bin = dfTest.copy()
encoder = ce.BinaryEncoder(cols=['model','transmission','fuelType'])
ford_bin = encoder.fit_transform(ford_bin)
ford_bin
def plotting_3_chart(df, feature):
    ## Importing seaborn, matplotlab and scipy modules. 
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import stats
    import matplotlib.style as style
    style.use('fivethirtyeight')

    ## Creating a customized chart. and giving in figsize and everything. 
    fig = plt.figure(constrained_layout=True, figsize=(12,8))
    ## creating a grid of 3 cols and 3 rows. 
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    #gs = fig3.add_gridspec(3, 3)

    ## Customizing the histogram grid. 
    ax1 = fig.add_subplot(grid[0, :2])
    ## Set the title. 
    ax1.set_title('Histogram')
    ## plot the histogram. 
    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)

    # customizing the QQ_plot. 
    ax2 = fig.add_subplot(grid[1, :2])
    ## Set the title. 
    ax2.set_title('QQ_plot')
    ## Plotting the QQ_Plot. 
    stats.probplot(df.loc[:,feature], plot = ax2)

    ## Customizing the Box Plot. 
    ax3 = fig.add_subplot(grid[:, 2])
    ## Set title. 
    ax3.set_title('Box Plot')
    ## Plotting the box plot. 
    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );
    
plotting_3_chart(ford_bin, 'price')
#skewness and kurtosis
print("Skewness: " + str(ford_bin['price'].skew()))
print("Kurtosis: " + str(ford_bin['price'].kurt()))
# There is a no improvment. Therefore, will continue with untransformed data.

## trainsforming target variable using numpy.log1p,
log_ford_bin = ford_bin.copy()
log_ford_bin["price"] = np.log1p(ford_bin["price"])

## Plotting the newly transformed response variable
plotting_3_chart(log_ford_bin, 'price')
#skewness and kurtosis
print("Skewness: " + str(log_ford_bin['price'].skew()))
print("Kurtosis: " + str(log_ford_bin['price'].kurt()))
# Registration year and mileage are most correlated with price, which is intuitive.

## Getting the correlation of all the features with target variable. 
(ford_bin.corr()**2)["price"].sort_values(ascending = False)[1:]
sns.set_style('whitegrid')
plt.subplots(figsize = (15,10))
## Plotting heatmap. 

# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(ford_bin.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


sns.heatmap(ford_bin.corr(), 
            cmap=sns.diverging_palette(20, 220, n=200), 
            mask = mask, 
            annot=True, 
            center = 0,
            fmt='.2f',
            linewidths=0.1,
            linecolor='white',
           );
## Give title. 
plt.title("Heatmap of all the Features", fontsize = 30);
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

reg = linear_model.LinearRegression()
X = ford_bin.drop('price', axis=1)
y = ford_bin['price']
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
reg.fit(X_train,y_train)
# decent performance.
reg.score(X_test,y_test)
results = X_test.copy()
results["predicted"] = reg.predict(X_test)
results["actual"]= y_test
results = results[['predicted', 'actual']]
results['predicted'] = results['predicted'].round(2)
results
ford.iloc[13981]
ford_onehot = ford.copy()
ford_onehot = pd.get_dummies(ford_onehot, columns=['model'])
ford_onehot = pd.get_dummies(ford_onehot, columns=['transmission'])
ford_onehot = pd.get_dummies(ford_onehot, columns=['fuelType'])

ford_onehot.head()
plotting_3_chart(ford_onehot, 'price')
#skewness and kurtosis
print("Skewness: " + str(ford_onehot['price'].skew()))
print("Kurtosis: " + str(ford_onehot['price'].kurt()))
## trainsforming target variable using numpy.log1p,
log_data = ford_onehot.copy()
log_data["price"] = np.log1p(log_data["price"])

## Plotting the newly transformed response variable
plotting_3_chart(log_data, 'price')
#skewness and kurtosis
print("Skewness: " + str(log_data['price'].skew()))
print("Kurtosis: " + str(log_data['price'].kurt()))
# Registration year and mileage are most correlated with price, which is intuitive.

## Getting the correlation of all the features with target variable. 
(log_data.corr()**2)["price"].sort_values(ascending = False)[1:]
from sklearn import linear_model
from sklearn.model_selection import train_test_split

reg = linear_model.LinearRegression()
X = log_data.drop('price', axis=1)
y = log_data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
reg.fit(X_train,y_train)
# Much improved score - one-hot encoding improves by 6% from previous model.
reg.score(X_test,y_test)
results = X_test.copy()
results["predicted"] = np.expm1(reg.predict(X_test))
results["actual"]= np.expm1(y_test)
results = results[['predicted', 'actual']]
results['predicted'] = results['predicted'].round(2)
results
custom = X_test.iloc[1].copy()
custom['year'] = 2017
custom['mileage'] = 16000
custom['fuelType_Diesel'] = 0
custom['fuelType_Petrol'] = 1
custom = custom.values.reshape(-1, 1)
flat_list = []
for sublist in custom:
    for item in sublist:
        flat_list.append(item)

print(np.expm1(reg.predict([flat_list])))