# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import matplotlib

import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import json

from pandas.io.json import json_normalize



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#just read data in json format. Probably the data came from a NoSQL database

X= pd.read_json('/kaggle/input/sao-paulo-real-estate-prediction/source-4-ds-train.json',orient='columns',lines=True)

X.head()

#some columns have nested data.Create a function to unnest

def unnest_column(df,nested_columns):

  for position in range(len(nested_columns)):

    column=nested_columns[position]

    un=df[column].apply(pd.Series)

    df=df.combine_first(un)

    #X=X.join(un,lsuffix='_left', rsuffix='_right')

    df=df.drop(columns=[column]) 

  return df



    

    

nested=['address','geoLocation','pricingInfos','location']

X=unnest_column(X,nested)



X.head()
#get rid of outliers

X=X.drop(X[(X['bathrooms']>6)].index)

X=X.drop(X[(X['bedrooms']>6)].index)

X=X.drop(X[(X['parkingSpaces']>6)].index)

X=X.drop(X[(X['suites']>5)].index)

X=X.drop(X[(X['totalAreas']>700) | (X['totalAreas']<28)].index)

X=X.drop(X[(X['usableAreas']>650) | (X['usableAreas']<28)].index)

X=X.drop(X[(X['price']>5000000) | (X['price']<100000)].index)





# yearlyIptu is the city tax. Varies from neighborhood and property size, so it's a good feature to predict price

X.yearlyIptu=X.yearlyIptu.replace(0,np.nan)

X.usableAreas=X.usableAreas.replace(0,np.nan)



#creates a column tax/size

X['paramIptu']=X.yearlyIptu/X.usableAreas



#group by neighborhood and extracts the mean 

X['paramIptu']=X.groupby('neighborhood').transform(lambda x: x.fillna(x.mean()))['paramIptu']



#fills the  missing data with the neighborhood mean , them multiply by size

X['estimatedIptu']=(X.usableAreas*X.paramIptu) 

X.yearlyIptu=X['yearlyIptu'].fillna(X['estimatedIptu'])



#some had to be filled with min value

X.yearlyIptu=X['yearlyIptu'].fillna(X.yearlyIptu.min())



#rental properties doesn't matter here

X=X[X.businessType !='RENTAL']





# select the features 

X_subset=X[[ 'parkingSpaces','lat','lon','suites','usableAreas','bathrooms','bedrooms','yearlyIptu','totalAreas','neighborhood']]

y=X['price']



#fill missing data

def feature_fill(df):

    df.totalAreas=df['totalAreas'].fillna(df['usableAreas'])

    df.usableAreas=df['usableAreas'].fillna(df['totalAreas'])

    df.lat=df['lat'].fillna(df['lat'].mean())

    df.lon=df['lon'].fillna(df['lon'].mean())

    df.suites=df['suites'].fillna(0)

    df.parkingSpaces=df['parkingSpaces'].fillna(0)

    df.bedrooms=df['bedrooms'].fillna(1)

    df.bathrooms=df['bathrooms'].fillna(1)

    df.bathrooms = df['bathrooms'].replace([0,0.0], 1)

    df.totalAreas=df['totalAreas'].fillna(df['totalAreas'].mean())

    df.usableAreas=df['usableAreas'].fillna(df['usableAreas'].mean())

    df=df[[ 'parkingSpaces','lat','lon','suites','usableAreas','bathrooms','bedrooms','yearlyIptu','neighborhood']]

    return df



X_subset=feature_fill(X_subset)

X_subset.isna().sum()
from sklearn.decomposition import PCA

from sklearn.preprocessing import normalize

from pandas.plotting import scatter_matrix

import matplotlib.cm as cm

from random import randint



X_data=X_subset[['parkingSpaces','lat','lon','suites','usableAreas','bathrooms','bedrooms','yearlyIptu']]

X_data2=normalize(X_data)

pca=PCA()

pca.fit(X_data2)

X_2=pca.transform(X_data2)





##Plot PCA

plt.plot(range(1,len(pca.components_)+1),pca.explained_variance_ratio_,'-o')

plt.xlabel('components')

plt.ylabel('% explained variance')

plt.title("Screen plot")

plt.show()

def find_relevant_variable(component, threshold):

    variables = []

    weights = []

    for i in range (0,len(component)):

        if abs(component[i])>threshold:

            variables.append(X_data.columns[i])

            weights.append(component[i])

    return variables, weights
print(find_relevant_variable(pca.components_[0],0.1)[0])

def retain_explanatory_components(pca, threshold):

    components = []

    importance = []

    

    for i in range (0,len(pca.components_)):

        if pca.explained_variance_[i]>threshold:

            components.append(pca.components_[i])

            importance.append(pca.explained_variance_ratio_[i])

    return components, importance
print(pca.components_[2], pca.explained_variance_ratio_[2])
X_original=X_subset[['parkingSpaces','lat','lon','suites','usableAreas','bathrooms','bedrooms','yearlyIptu','neighborhood']]

X_simplified=X_subset[['usableAreas','lat','lon','yearlyIptu','neighborhood']]

from sklearn import metrics, svm

from sklearn.metrics import mean_squared_error,  mean_absolute_error, r2_score

from sklearn.linear_model import Lasso, ElasticNetCV, SGDRegressor, LinearRegression, Ridge

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsRegressor

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_simplified, y, test_size=0.25, random_state=42)

#grid_params ={'n_neighbors':[9,10,15,20],

#              'weights':['uniform','distance'],

#              'metric':['euclidean','manhattan']}

#neigh = GridSearchCV(KNeighborsRegressor(),grid_params,verbose=1,cv=3,n_jobs=8)





#neigh=KNeighborsRegressor(metric='manhattan', n_neighbors=12, weights='distance', n_jobs=8)

#neigh.fit(X_train, y_train)

#y_pred=neigh.predict(X_test)



#did lasso and elastic net regression, but with poor results.

#lasso can return the most important features in prediction



#lasso=Lasso(alpha=0.0001)

#lasso.fit(X_train,y_train)

#y_pred=lasso.predict(X_test)





#numeric_features = ['parkingSpaces','lat','lon','suites','usableAreas','bathrooms','bedrooms','yearlyIptu']

numeric_features = ['usableAreas','lat','lon','yearlyIptu']

numeric_transformer = Pipeline(steps=[(('scaler', StandardScaler())),])

categorical_features = ['neighborhood']

categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])



preprocessor = ColumnTransformer(transformers=[

     ('cat', categorical_transformer, categorical_features),

        ('num', numeric_transformer, numeric_features)])



neigh= Pipeline(steps=[('preprocessor', preprocessor),

                      ('classifier', KNeighborsRegressor( leaf_size=30, metric='manhattan', n_neighbors=14, weights='distance', n_jobs=8))])



neigh.fit(X_train, y_train)

y_pred=neigh.predict(X_test)



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('R2 Score:', r2_score(y_test,y_pred)) 

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Heteroscedasticity

errors=y_test-y_pred

plt.scatter(X_test['usableAreas'],errors,color='b')

#just the code to predict the prices on test data 

X_pred= pd.read_json('/kaggle/input/real-estate-prediction-challenge/source-4-ds-test.json',orient='columns',lines=True)

X_pred=unnest_column(X_pred,nested)

X_pred=feature_arrange(X_pred).fillna(0)

y_pred_test=neigh.predict(X_pred)

y_pred_test