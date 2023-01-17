# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



df = pd.read_csv("../input/kc_house_data.csv")

df

#df.isnull()

#df.dtypes

#df.corr()

#df[df.waterfront ==1]

#find out the unique value in each column

#column_list = df.columns.values.tolist()

#print(column_list)

#for column_name in column_list:

    #print(column_name,len(df[column_name].unique()))

    

#code to create dataframe of unique values



UnqiueCountDF = pd.DataFrame(columns=['feature', 'UniqueValues'])

for column in df.columns:

    UnqiueCountDF = UnqiueCountDF.append({'feature': column, 'UniqueValues': len(df[column].unique())},ignore_index=True)

    



df.corr()



#df.drop(["id"],axis=1,inplace = True)



#len(df.waterfront.unique()) 



#regession plot

import seaborn as sns

%matplotlib inline



feature_cols = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat']





columns_scatterplot=UnqiueCountDF[UnqiueCountDF['UniqueValues']>70]['feature'].values

columns_boxplot=UnqiueCountDF[UnqiueCountDF['UniqueValues']<=70]['feature'].values



import matplotlib.pyplot as plt



for feature in columns_scatterplot:

    fig = plt.figure()

    fig.suptitle(feature, fontsize=20)

    plt.scatter(df[feature], df['price'])

    plt.show()





import seaborn as sns

for feature in columns_boxplot:

    ax = sns.boxplot(x=feature, y='price', data=df)

    plt.show()

    

#x = df[feature.cols]

#y = df['price'].values



#feature engineering

#x['basement_present'] = x[]



import matplotlib.pyplot as plt

h = df[feature_cols].hist(bins=10,figsize=(16,16))

df.head(9)

#correlation heatmap

import matplotlib.pyplot as plt

def correlation_heatmap(df1):

    _, ax = plt.subplots(figsize = (15, 10))

    colormap= sns.diverging_palette(110, 10,as_cmap = True)

    sns.heatmap(df.corr(), annot=True, cmap = colormap)



correlation_heatmap(df)



plt.figure(figsize=(12,5))

sns.distplot(df['price'])



#creating training and testing dataset



from sklearn import preprocessing,decomposition

feature_cols = ['date','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode']



x=df[feature_cols]

y=df['price'].values



#Feature Engineering

x['basement_present'] = x['sqft_basement'].apply(lambda x: 1 if x > 0 else 0) # Indicate whether there is a basement or not

x['renovated'] = x['yr_renovated'].apply(lambda x: 1 if x > 0 else 0) # 1 if the house has been renovated



x['sales_yr']=x['date'].astype(str).str[:4]

x['HouseAge'] =  x['sales_yr'].astype(int) - x['yr_built']

x['age_rnv']=0

x['age_rnv']=x['sales_yr'][x['yr_renovated']!=0].astype(int)-x['yr_renovated'][x['yr_renovated']!=0]

x['age_rnv'][x['age_rnv'].isnull()]=0



#partition age into bins

bins = [-2,0,5,10,25,50,75,100,100000]

labels = ['<1','1-5','6-10','11-25','26-50','51-75','76-100','>100']

x['age_binned'] = pd.cut(x['HouseAge'], bins=bins, labels=labels)

# partition the age_rnv into bins

bins = [-2,0,5,10,25,50,75,100000]

labels = ['<1','1-5','6-10','11-25','26-50','51-75','>75']

x['age_rnv_binned'] = pd.cut(x['age_rnv'], bins=bins, labels=labels)



x = x.drop(['date'],axis=1)



x = pd.get_dummies(x, prefix='Category_', columns=['age_rnv_binned','age_binned','view','condition','zipcode'])

print(x.columns)

x = preprocessing.normalize(x)



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.3, random_state = 101)



from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn import metrics

from sklearn.preprocessing import PolynomialFeatures

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR



def fitmodel(x_train,y_train,x_test,y_test):

    regressor=LinearRegression()

    regressor.fit(x_train, y_train)

    linear_predicted_Values=regressor.predict(x_test)

    print('Linear Regression R-squared', round(regressor.score(x_test, y_test), 3)) 

    

    regressor=Lasso(alpha=1)

    regressor.fit(x_train, y_train)

    ridge_predicted_Values=regressor.predict(x_test)

    print('Lasso Regression R-squared', round(regressor.score(x_test, y_test), 3))

    

    regressor=Ridge(alpha=1)

    regressor.fit(x_train, y_train)

    ridge_predicted_Values=regressor.predict(x_test)

    print('Ridge Regression R-squared', round(regressor.score(x_test, y_test), 3))

    

    '''

    poly = PolynomialFeatures(degree=2)

    x_train_ = poly.fit_transform(x_train)

    x_test_ = poly.fit_transform(x_test)

    

    regressor = LinearRegression()

    regressor.fit(x_train_, y_train)

    Poly_Predicted_Values=regressor.predict(x_test_)

    print('polynomial Regression R-squared', round(regressor.score(x_test_, y_test), 3))

    '''

    '''

    regressor = SVR()

    regressor.fit(x_train, y_train)

    SVR_predicted_Values=regressor.predict(x_test)

    print('SVR Regression R-squared', round(regressor.score(x_test, y_test), 3))

    '''

    

    regressor = GradientBoostingRegressor(n_estimators=500,max_depth=4,min_samples_split=2,learning_rate=0.05,loss='ls')

    regressor.fit(x_train, y_train)

    GBM_predicted_Values=regressor.predict(x_test)

    print('Gradient Boosting Regression R-squared', round(regressor.score(x_test, y_test), 3))

    

        

fitmodel(x_train,y_train,x_test,y_test)



#checking accuracy matrices

regressor = GradientBoostingRegressor(n_estimators=200,max_depth=4,min_samples_split=2,learning_rate=0.05,loss='ls')

regressor.fit(x_train, y_train)

GBM_predicted_Values=regressor.predict(x_test)

print('Gradient Boosting Regression R-squared', round(regressor.score(x_test, y_test), 3))

mean_squared_error = metrics.mean_squared_error(y_test, GBM_predicted_Values)

print('Mean Squared Error (MSE) ', round(np.sqrt(mean_squared_error), 2)) 

print('R-squared (training) ', round(regressor.score(x_train, y_train), 3))

print('R-squared (testing) ', round(regressor.score(x_test, y_test), 3)) 



from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor()

grid_values = {'n_estimators': [100, 200,300],'max_depth': [2,3,4],'min_samples_split':[2,3,4],'min_samples_leaf':[2,3,4]}

grid_gbm = GridSearchCV(estimator=clf, param_grid = grid_values, cv= 3,n_jobs=-1)



grid_gbm.fit(x_train, y_train)



print('Accuracy Score : ' + str(grid_gbm.best_score_))



# Any results you write to the current directory are saved as output.