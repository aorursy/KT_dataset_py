import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Read data
# Convert text to NaN
data = pd.read_excel('../input/6370-voc-data/6370GD.xlsx', na_values = '[-11059] No Good Data For Calculation')
data.head()
# Select top 3 rows as column names for reference later
col_names = data.iloc[:3,3:37]
col_names

# Define df to include header and data only
df = data.iloc[2:,3:37]
df.head()
# Select top row as header
# Remove duplicated row
# Reset row index
# Print shape of df
df.columns = df.iloc[0]
df = df.drop(2)
df = df.reset_index(drop=True)
df.head()
# Apply isna for every row
# Print number of rows with NaN
df.isna()
# Drop row if entire rows contain Na
df = df.dropna(how='all')
df
# Check rows containing Na
df.isna().sum()
df.info()
# Drop rows containing at least 1 Na
df = df.dropna()
df
df.isna().sum()
df.info()

# Convert all datatype from object to float64
df = df.astype('float64')
df.info()
# Define y1 as inlet LEL and y2 as outlet LEL
y1 = df.iloc[:,0]
y2 = df.iloc[:,1]
y1,y2
# Define X
X = df.iloc[:,2:]
X
# Split data into training and testing datasets
from sklearn.model_selection import train_test_split

X1_train, X1_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.3, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.3, random_state=42)
print( "X1:", X1_train.shape, y1_train.shape, "X2:", X2_train.shape, y2_train.shape)

# Standardize data to have centered mean and unit standard deviation
from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler().fit(X1_train)
scaler2 = StandardScaler().fit(X2_train)
X1_train_s = scaler1.transform(X1_train)
X2_train_s = scaler2.transform(X2_train)
X1_test_s = scaler1.fit_transform(X1_test)
X2_test_s = scaler2.fit_transform(X2_test)
print("Mean of train1", round(X1_train_s.mean()),"Stdev of train1", X1_train_s.std(), "\nMean of test1", round(X1_test_s.mean()), "Stdev of test1", X1_test_s.std())
print("Mean of train2", round(X2_train_s.mean()),"Stdev of train2", X2_train_s.std(), "\nMean of test2", round(X2_test_s.mean()), "Stdev of test2", X2_test_s.std())
# We use simple Linear Regression model to fit the training set. There will be two sets of modelling, one for y1 and y2
# Once the model is trained, it is used to predict the outcome of the test data set, which it has not seen
# The score shows R-squared value of test dataset, the higher the btter
# The Mean Squared Error and Root Mean Square Error shows the error between test and predicted outcomes, the lower the better

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


linreg1 = LinearRegression().fit(X1_train_s, y1_train)
linreg2 = LinearRegression().fit(X2_train_s, y2_train)
y1_pred = linreg1.predict(X1_test_s)
y2_pred = linreg2.predict(X2_test_s)
score1 = linreg1.score(X1_test_s, y1_test)
score2 = linreg2.score(X2_test_s, y2_test)
mse1 = mean_squared_error(y1_test, y1_pred)
mse2 = mean_squared_error(y2_test, y2_pred)

print("LinearRegression LEL inlet", "R2", round(score1,2), "MSE", round(mse1,2))
print("LinearRegression LEL outlet", "R2", round(score2,2), "MSE", round(mse2,2))
# The plot shows how well our model predict the outcome which is the LEL% from the test data set
# The model has not been trained on the test set so the prediction is unbiased 

x_ax = range(len(X1_test_s))
plt.scatter(x_ax, y2_test, s=5, color="blue", label="y2 outcome")
plt.plot(x_ax, y2_pred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()
# Define X_names only to include X variable columns and transpose it from horizontal to vertical
# Define coef to be a vertical dataframe of regression model coefficients
X_names = col_names.iloc[:,2:]
X_names.columns = range(X_names.shape[1])
linreg_importance = X_names.T
linreg_importance.columns = ['desc','unit','tag']

coef = pd.DataFrame(linreg2.coef_)
linreg_importance['coefficient'] = coef
linreg_importance['rank'] = linreg_importance['coefficient'].abs()
linreg_importance = linreg_importance.sort_values(by=['rank'], ascending = False)
linreg_importance['rank'] = range(1,len(linreg_importance.unit)+1)
linreg_importance.head(10)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

DTR1 = DecisionTreeRegressor()
DTR1.fit(X1_train, y1_train)
score1 = DTR1.score(X1_test, y1_test)
feature1 = DTR1.feature_importances_

DTR2 = DecisionTreeRegressor()
DTR2.fit(X2_train, y2_train)
score2 = DTR.score(X2_test, y2_test)
feature2 = DTR2.feature_importances_

print("R2 for y1:", score1, "\nR2 for y2:", score2)
feature1, feature2
DTR_importance = X_names.T
DTR_importance.columns = ['desc','unit','tag']

coef = pd.DataFrame(feature1)
DTR_importance['coefficient'] = coef
DTR_importance['rank'] = DTR_importance['coefficient'].abs()
DTR_importance = DTR_importance.sort_values(by=['rank'], ascending = False)
DTR_importance['rank'] = range(1,len(DTR_importance.unit)+1)
DTR_importance.head(10)

DTR_importance
coef2 = pd.DataFrame(feature2)
DTR_importance['coefficient'] = coef2
DTR_importance['rank'] = DTR_importance['coefficient'].abs()
DTR_importance = DTR_importance.sort_values(by=['rank'], ascending = False)
DTR_importance['rank'] = range(1,len(DTR_importance.unit)+1)
DTR_importance.head(10)

DTR_importance