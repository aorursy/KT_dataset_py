# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import os

# to save model

import pickle

# Import visualization modules

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import os

#Step 2 : Data import

# Use pandas to read in csv file

New=pd.read_csv(('/kaggle/input/housesalesprediction/kc_house_data.csv'))

print(New)
New.describe()
New.dtypes
missing_values = New.isnull()

missing_values.head(5)
# Visualize the data

# Use seaborn to conduct heatmap to identify missing data

# data -> argument refers to the data to creat heatmap

# yticklabels -> argument avoids plotting the column names

# cbar -> argument identifies if a colorbar is required or not

# cmap -> argument identifies the color of the heatmap

sns.heatmap(data = missing_values, yticklabels=False, cbar=False, cmap='viridis')
#find the pearson correlation : which attribute is highly correlated
features = ['price','bedrooms','bathrooms','sqft_lot','floors','waterfront',

            'view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated',

            'zipcode','lat','long','sqft_living15','sqft_lot15','sqft_living']



mask = np.zeros_like(New[features].corr(), dtype=np.bool) 

mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))

plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(New[features].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", 

            #"BuGn_r" to reverse 

            linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});
# Split data into 'X' features and 'y' target label sets

X = New[['sqft_above']]

y = New['sqft_living']
from sklearn import linear_model

from sklearn import metrics

from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
evaluation = pd.DataFrame({'Model': [],

                           'Details':[],

                           'Root Mean Squared Error (RMSE)':[],

                           'R-squared (training)':[],

                           'Adjusted R-squared (training)':[],

                           'R-squared (test)':[],

                           'Adjusted R-squared (test)':[]})

train_data,test_data = train_test_split(New,train_size = 0.75,random_state=20)



lr = linear_model.LinearRegression()

X_train = np.array(train_data['sqft_above'], dtype=pd.Series).reshape(-1,1)

y_train = np.array(train_data['sqft_living'], dtype=pd.Series)

lr.fit(X_train,y_train)



X_test = np.array(test_data['sqft_above'], dtype=pd.Series).reshape(-1,1)

y_test = np.array(test_data['sqft_living'], dtype=pd.Series)



pred = lr.predict(X_test)

rmsesm = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))

rtrsm = float(format(lr.score(X_train, y_train),'.3f'))

rtesm = float(format(lr.score(X_test, y_test),'.3f'))

print ("Average Price for Test Data: {:.3f}".format(y_test.mean()))

print('Intercept: {}'.format(lr.intercept_))

print('Coefficient: {}'.format(lr.coef_))



r = evaluation.shape[0]

evaluation.loc[r] = ['Simple Linear Regression','-',rmsesm,rtrsm,'-',rtesm,'-']

evaluation
import pickle

from sklearn.linear_model import LinearRegression

#Step 7.0 Save the model in pickle

#Save to file in the current working directory

pkl_filename = "pickle_model.pkl"

with open(pkl_filename, 'wb') as file:

    pickle.dump(lr, file)

    # Load from file

with open(pkl_filename, 'rb') as file:

    pickle_model = pickle.load(file)
# Calculate the accuracy score and predict target values

score = pickle_model.score(X_test,y_test)

Ypredict = pickle_model.predict(X_test)

print("Test score: {0:.2f} %".format(100 * score))
df = pd.DataFrame({'Actual': y_test, 'Predicted': Ypredict.flatten()})

df
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, Ypredict))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, Ypredict))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Ypredict)))
# plotting regression line

ax = plt.axes()

ax.scatter(X, y)

plt.title("Input Data and regression line ") 

ax.plot(X_test, Ypredict, color ='Red')

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.axis('tight')

plt.show()
os.path.isfile('/kaggle/input/housesalesprediction/kc_house_data.csv')

run = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')

print('data import')

print(run.head(20))
Newpredict = pickle_model.predict(run[['X']])

output=run[['Y']]

output['Y_Predicted']=Ypredict

output

#additional random forest
from sklearn.ensemble import RandomForestRegressor

rf_regressor=RandomForestRegressor(n_estimators=28, random_state=0)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import explained_variance_score

rf_regressor.fit(X_train,y_train)

rf_regressor.score(X_test,y_test)

rf_pred=rf_regressor.predict(X_test)

rf_score=rf_regressor.score(X_test,y_test)

expl_rf=explained_variance_score(rf_pred,y_test)
print("Random Forest regression Model Score is",round(rf_regressor.score(X_test,y_test)*100))
#third estimators (using multiple factors ) - according to the pearson correlation model consider highly correlated (0.50 )
# Split data into 'X' features and 'y' target label sets

X1 = New[['sqft_above','sqft_living15','bathrooms','grade','price','bedrooms','view','floors','sqft_basement']]

y1 = New['sqft_living']
# Import module to split dataset

from sklearn.model_selection import train_test_split

# Split data set into training and test sets

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.25, random_state=100)
model = LinearRegression()

output_model=model.fit(X1_train, y1_train)

output_model

pkl_filename = "pickle_model.pkl"

with open(pkl_filename, 'wb') as file:

    pickle.dump(model, file)



# Load from file

with open(pkl_filename, 'rb') as file:

    pickle_model = pickle.load(file)



# Calculate the accuracy score and predict target values

score = pickle_model.score(X1_test, y1_test)

print("Test score: {0:.2f} %".format(100 * score))

Ypredict = pickle_model.predict(X1_test)
df = pd.DataFrame({'Actual': y1_test, 'Predicted': Ypredict.flatten()})

df