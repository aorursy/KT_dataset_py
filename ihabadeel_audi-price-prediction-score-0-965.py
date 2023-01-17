#Importing necessary modules



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set_style('whitegrid')
df = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/audi.csv')
df.head()
df.info()
df.describe()
#Checking the models to check for feature engineering 



df.model.unique()
#Creating two columns for Series and Series Model



def Series(model):

    series = ''

    for char in model:

        if char.isalpha():

            series += char

    return series



def Smodel(model):

    smodel = ''

    for char in model:

        if char.isdigit():

            smodel += char

    if smodel == '':

        return 1

    return int(smodel)
df['series'] = df['model'].apply(Series)

df['smodel'] = df['model'].apply(Smodel)
#Adding a column as age of the car and drooping the year column



df['age'] = df['year'].apply(lambda x: 2020-x)

df.drop('year', axis=1, inplace=True)
df.head()
plt.figure(figsize=(20,8))



sns.countplot(y=df['model'])
fig = plt.figure(figsize=(18,6))



fig.add_subplot(1,2,1)

sns.countplot(df['transmission'])

fig.add_subplot(1,2,2)

sns.countplot(df['fuelType'])
sns.pairplot(df)
plt.figure(figsize=(12,6))



sns.countplot(df['series'])
num_cols = df.select_dtypes(exclude=['object'])



fig = plt.figure(figsize=(20,8))



for col in range(len(num_cols.columns)):

    fig.add_subplot(2,4,col+1)

    sns.distplot(num_cols.iloc[:,col], hist=False, rug=True, kde_kws={'bw':0.1}, label='UV')

    plt.xlabel(num_cols.columns[col])



plt.tight_layout()
fig = plt.figure(figsize=(20,8))

plt.title('VS. Price')



for col in range(len(num_cols.columns)):

    fig.add_subplot(2,4,col+1)

    sns.scatterplot(x=num_cols.iloc[:,col], y=df['price'], label='MV')

    plt.xlabel(num_cols.columns[col])

    plt.ylabel('Price')



plt.tight_layout()
plt.figure(figsize=(12,8))

sns.heatmap(df.corr(), annot=True, cmap='Blues', linewidth=0.5)
#Dropping the models columns and one hot encoding the rest



df.drop('model', axis=1, inplace=True)



df = pd.get_dummies(df)

df.head()
#importing all the necessary modules for ML



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.preprocessing import StandardScaler
#Scaling the data



scaler = StandardScaler()

scaled = scaler.fit_transform(df)

df = pd.DataFrame(scaled, columns=df.columns)



df.head()
#Generating the test and train datasets



X = df.drop('price', axis=1)

y = df['price']



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
#Making a list of Models to test against



models = [['Linear Regression',LinearRegression()],

         ['Decision Tree',DecisionTreeRegressor()],

         ['Random Forest',RandomForestRegressor(n_estimators=100,n_jobs=-1)],

         ['XGBoost',XGBRegressor(learning_rate=0.05,n_jobs=-1,n_estimators=1000)]]
#Evaluating the models



for name, model in models:

    model.fit(X_train,y_train)

    predictions = model.predict(X_test)

    

    print('{} Score: '.format(name), model.score(X_test,y_test))

    print('{} MAE: '.format(name), mean_absolute_error(y_test,predictions))

    print('{} MSE: '.format(name), mean_squared_error(y_test,predictions))

    print('{} RMSE: '.format(name), np.sqrt(mean_squared_error(y_test,predictions)), end='\n\n')
#Since XGB gave the best performance, we'll use it for the final predictions



boost = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1)



boost.fit(X_train,y_train,early_stopping_rounds=5, eval_set=[(X_test,y_test)], verbose=False)

pred = boost.predict(X_test)



print('Boost Score: ', model.score(X_test,y_test))

print('Boost MAE: ', mean_absolute_error(y_test,predictions))

print('Boost MSE: ', mean_squared_error(y_test,predictions))

print('Boost RMSE: ', np.sqrt(mean_squared_error(y_test,predictions)), end='\n\n')
#Scatterplot showing the spread of actual and predicted prices



sns.scatterplot(x=pred, y=y_test)



plt.title('Predicted vs Actual Price')

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')