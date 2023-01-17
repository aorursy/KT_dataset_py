import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from scipy import stats



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

directory = "../input/car-purchase-data/"

feature_tables = ['Car_Purchasing_Data.csv']



df_train = directory + feature_tables[0]



# Create dataframes

print(f'Reading csv from {df_train}...')

train = pd.read_csv(df_train,encoding='ISO-8859-1')

print('...Complete')
train.head()
train = train.drop(['Customer Name','Customer e-mail', 'Country'],axis=1)
corr = train.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(20, 18))



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, annot=True, mask=mask, cmap="YlGnBu", center=0,

            square=True, linewidths=.5)
features = ['Age', 'Annual Salary', 'Net Worth']

target = ['Car Purchase Amount']



X = train[features]

y = train[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(regressor.coef_)

print(regressor.intercept_)
print(f'Linear Regression equation is Y = {round(regressor.coef_[0][0],3)}*X0 + {round(regressor.coef_[0][1],3)}*X1 + {round(regressor.coef_[0][2],3)}*X2 {round(regressor.intercept_[0],3)}'  )

print(f'Y = {target[0]}')

print(f'X0 = {features[0]}')

print(f'X1 = {features[1]}')

print(f'X2 = {features[2]}')
r2 = metrics.r2_score(y_test,y_pred)

mae = metrics.mean_absolute_error(y_test,y_pred)

mse = metrics.mean_squared_error(y_test,y_pred)

print(f'R2 = {r2}')

print(f'MAE = {mae}')

print(f'RMSE = {mse}')
age = 49

salary = 60500

net_worth = 173000

customer = [age, salary, net_worth]

prediction = regressor.predict([customer])
print('--Customer Information--')

print(f'Age: {age}') 

print(f'Salary: {salary}')

print(f'Net Worth: {net_worth}')

print()

print(f'Predicted Car Purchase Amount: {round(prediction[0][0],2)}')