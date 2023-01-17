

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



df = pd.read_json("amsterdam.json")

df.info()

import pandas_profiling as pp



from pandas_profiling import ProfileReport



prof = ProfileReport(df)



prof.to_file(output_file="profrep.html")



df = df.dropna()



df.head()



df['price']=df['price'].apply(lambda x : x.replace('$',''))

df['price']=df['price'].apply(lambda x : x.replace(',',''))

df['price'] = pd.to_numeric(df['price'])



#Create heatmap to visualize



amsterdam = {'lat' : 52.3667, 'long' : 4.8945}



import folium



m=folium.Map(location =[52.3667,4.8945],zoom_start=13)



from folium import plugins







heatmap = df[['latitude', 'longitude', 'price']].round(4).groupby(['latitude', 'longitude']).sum().reset_index().values.tolist()



# plot heatmap

m.add_child(plugins.HeatMap(heatmap, radius=9, max_zoom=10))

m

m.save('index.html')





df=pd.get_dummies(df)





X = df.drop(['price'], axis=1)



y = df['price']





from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=0.3, random_state=123)



#1. LINEAR REGRESSION



from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)



from sklearn import metrics

r2 = metrics.r2_score(y_test, y_pred)

mae = metrics.mean_absolute_error(y_test, y_pred)

scores = pd.DataFrame({'Baseline (regression)' : [r2, mae]}, index=['R2', 'MAE'])

scores



#2. SVM



import os

#if 'svr_gridsearch_cv.pkl' in os.listdir():

    

 #   svr_grid_search = joblib.load('svr_gridsearch_cv.pkl')

    

#else:



from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV

svr = SVR()

param_grid = [

      {'C': [1, 10, 100, 1000], 'kernel': ['linear']},

      {'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf']}]

svr_grid_search = GridSearchCV(svr, param_grid=param_grid, 

                                   n_jobs=-1, 

                                   scoring=['r2','neg_mean_squared_error'],

                                  refit='neg_mean_squared_error', verbose=100)

svr_grid_search.fit(X_train, y_train)

#joblib.dump(svr_grid_search.best_estimator_, 'svr_gridsearch_cv.pkl')









#3. LighGBM Regressor



from lightgbm import LGBMRegressor

  

reg = LGBMRegressor()



reg.fit(X_train,y_train)



from sklearn import metrics

metrics.mean_squared_error(y_test,reg.predict(X_test))



imp_feat=pd.Series(reg.feature_importances_,index=X.columns.tolist())

imp_feat.sort_values(ascending=False).plot(kind='bar', )






