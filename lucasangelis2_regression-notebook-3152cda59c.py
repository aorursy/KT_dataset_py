# Import libraries
import pandas as pd
import sklearn
import numpy as np
import warnings
# Training database reading
trainDB = pd.read_csv("../input/train.csv",
                     sep=r'\s*,\s*',
                     engine='python',
                     na_values="NaN")
# Testing database reading
testDB = pd.read_csv("../input/test.csv",
                     sep=r'\s*,\s*',
                     engine='python',
                     na_values="NaN")
# Train database visualization
trainDB.iloc[0:20,:]
# Train database size
size_train = trainDB.shape
size_train
# Plot library
import matplotlib.pyplot as plt
# Total Rooms analysis

total_r = trainDB['total_rooms'] 
r_max = total_r.max()
r_min = total_r.min()
print('O maximo de cômodos em uma regiao eh',r_max ,'e o mínimo eh ',r_min)

# Classes division -> 0 - 1k, 1k - 5k, 5k - 10k, 10k - 40k
clas = np.array([0,0,0,0])
for i in range(len(total_r)):
    if total_r[i] <= 1000:
        clas[0] = clas[0] + 1
    if total_r[i] > 1000 and total_r[i] <= 5000:
        clas[1] = clas[1] + 1
    if total_r[i] > 5000 and total_r[i] <= 10000:
        clas[2] = clas[2] + 1
    if total_r[i] > 10000:
        clas[3] = clas[3] + 1
lab = '0-1k','1k<X<5k','5k<X<10k','10k<X<40k'
plt.pie(clas,labels = lab);
# The same for Total bedrooms

total_b = trainDB['total_bedrooms'] 
r_max = total_b.max()
r_min = total_b.min()
print('O maximo de quartos em uma regiao eh',r_max,'e o mínimo eh ',r_min)

# Classes division -> 0 - 500, 500 - 1k, 1k - 2.5k, 2.5k - Max
clas = np.array([0,0,0,0])
for i in range(len(total_r)):
    if total_b[i] <= 500:
        clas[0] = clas[0] + 1
    if total_b[i] > 500 and total_b[i] <= 1000:
        clas[1] = clas[1] + 1
    if total_b[i] > 1000 and total_b[i] <= 2500:
        clas[2] = clas[2] + 1
    if total_b[i] > 2500:
        clas[3] = clas[3] + 1
lab = '0-500','500<X<1k','1k<X<2.5k','2.5k<X<Max'
plt.pie(clas,labels = lab);
# (Total bedrooms) / (Total rooms)

rel_rooms_bedrooms = total_r/total_b
rel_max = rel_rooms_bedrooms.max()
rel_min = rel_rooms_bedrooms.min()
print('Max = ',rel_max,' Min = ',rel_min)
values = trainDB['median_house_value']
# Scatter to look for Correlation
plt.scatter(rel_rooms_bedrooms,values);
# Relation Median Income - Median House Value
med_in = trainDB['median_income']
# Scatter to look for Correlation
plt.scatter(med_in,values);
# per capita income related to median house value
pc_income = med_in/trainDB['population']
# Scatter to look for Correlation
plt.scatter(pc_income,values);
# Taking X and Y for the regressors
size_train = trainDB.shape
# Testing best columns to use in knn regressor
knn_trainX = trainDB[["longitude","total_rooms","total_bedrooms","population","households","median_income"]]
# Testing best columns to use in lasso regressor
lasso_trainX = trainDB[["longitude","median_age","total_rooms","total_bedrooms","population","households","median_income"]]
# Testing best columns to use in ridge regressor
ridge_trainX = trainDB[["longitude","median_age","total_rooms","total_bedrooms","population","households","median_income"]]
# Label
trainY = trainDB.iloc[:,(size_train[1])-1]
# Metrics
from sklearn.metrics import mean_squared_error as mse
import math
def rmsle(y, y_pred):
	assert len(y) == len(y_pred)
	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
# KNN Regression
# importing regressor
from sklearn.neighbors import KNeighborsRegressor
# Defining neighborhood
neigh = KNeighborsRegressor(n_neighbors=2)
# Fitting
neigh.fit(knn_trainX,trainY) 
# Predict
knn_predict = neigh.predict(knn_trainX)
df_knn = pd.DataFrame({'Y_real':trainY[:],'Y_pred':knn_predict[:]})
# RMLSE (Estimativa do desempenho do regressor)
print(rmsle(df_knn.Y_real,df_knn.Y_pred))
# LASSO Regression 
warnings.filterwarnings("ignore")
# importing regressor
from sklearn import linear_model
# defining regressor
clf = linear_model.Lasso(alpha=0.5)
# fitting to data
clf.fit(lasso_trainX,trainY)
# predicting
lasso_predict = clf.predict(lasso_trainX)
df_l = pd.DataFrame({'Y_real':trainY[:],'Y_pred':lasso_predict[:]})
# inverting negative numbers
for i in range (len(df_l.Y_real)):
    if df_l.Y_pred[i] < 0:
        aux = df_l.Y_pred[i]*(-1)
        df_l.ix[i,'Y_pred'] = aux
# RMSLE (Estimativa do desempenho do regressor)
print(rmsle(df_l.Y_real,df_l.Y_pred))
# RIDGE Regression
# importing regressor
from sklearn.linear_model import Ridge
# defining regressor
clf = Ridge(alpha=1.0)
# fitting to data
clf.fit(ridge_trainX,trainY) 
# predicting
ridge_predict = clf.predict(ridge_trainX)
df_r = pd.DataFrame({'Y_real':trainY[:],'Y_pred':ridge_predict[:]})
for i in range (len(df_r.Y_real)):
    if df_r.Y_pred[i] < 0:
        aux = df_r.Y_pred[i]*(-1)
        df_r.ix[i,'Y_pred'] = aux
# RMSLE (Estimativa do desempenho do regressor)
print(rmsle(df_r.Y_real,df_r.Y_pred))
# Inserting new features 
new_trainX = trainDB.iloc[:,1:9]
# per capita income
new_trainX.insert(5,'per_capita_income',pc_income,allow_duplicates=False)
# (Total bedrooms) / (Total rooms)
new_trainX.insert(5,'total_rooms_total_bedrooms',rel_rooms_bedrooms,allow_duplicates=False)
# people per house
p_p_h = new_trainX.population/new_trainX.households
new_trainX.insert(5,'people_per_house',p_p_h,allow_duplicates=False)
new_trainX.iloc[0:10,:]
# Import Select K Best and f_classif
from sklearn.feature_selection import SelectKBest,f_regression
# Import kNN classifier
from sklearn.neighbors import KNeighborsClassifier
# Defining selector for knn
selector = SelectKBest(score_func=f_regression, k=8)
# Training the selector
trainX_select = selector.fit_transform(new_trainX,trainY)
ids_knn = selector.get_support(indices = True)
knn_trainX_trans = new_trainX.iloc[:,ids_knn]
#KNN
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(knn_trainX_trans,trainY) 
knn_predict = neigh.predict(knn_trainX_trans)
df_knn = pd.DataFrame({'Y_real':trainY[:],'Y_pred':knn_predict[:]})
# Estimativa do desempenho do regressor
print(rmsle(df_knn.Y_real,df_knn.Y_pred))
# Defining selector for Lasso
selector = SelectKBest(score_func=f_regression, k=8)
# Training the selector
trainX_select = selector.fit_transform(new_trainX,trainY)
ids = selector.get_support(indices = True)
lasso_trainX_trans = new_trainX.iloc[:,ids]
#Lasso
warnings.filterwarnings("ignore")
from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.5)
clf.fit(lasso_trainX_trans,trainY)
lasso_predict = clf.predict(lasso_trainX_trans)
df_l = pd.DataFrame({'Y_real':trainY[:],'Y_pred':lasso_predict[:]})
for i in range (len(df_l.Y_real)):
    if df_l.Y_pred[i] < 0:
        aux = df_l.Y_pred[i]*(-1)
        df_l.ix[i,'Y_pred'] = aux
# Estimativa do desempenho do regressor
print(rmsle(df_l.Y_real,df_l.Y_pred))
# Defining selector for Ridge
selector = SelectKBest(score_func=f_regression, k=8)
# Training the selector
trainX_select = selector.fit_transform(new_trainX,trainY)
ids = selector.get_support(indices = True)
ridge_trainX_trans = new_trainX.iloc[:,ids]
# RIDGE
from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0)
clf.fit(ridge_trainX_trans,trainY) 
ridge_predict = clf.predict(ridge_trainX_trans)
df_r = pd.DataFrame({'Y_real':trainY[:],'Y_pred':ridge_predict[:]})
for i in range (len(df_r.Y_real)):
    if df_r.Y_pred[i] < 0:
        aux = df_r.Y_pred[i]*(-1)
        df_r.ix[i,'Y_pred'] = aux
# Estimativa do desempenho do regressor
print(rmsle(df_r.Y_real,df_r.Y_pred))
# Submission
testX = testDB.iloc[:,1:9]
# per capita income
pc_income = testX.median_income/testX.population
testX.insert(5,'per_capita_income',pc_income,allow_duplicates=False)
# (Total bedrooms) / (Total rooms)
rel_rooms_bedrooms = testX.total_bedrooms/testX.total_rooms
testX.insert(5,'total_rooms_total_bedrooms',rel_rooms_bedrooms,allow_duplicates=False)
# people per house
p_p_h = testX.population/testX.households
testX.insert(5,'people_per_house',p_p_h,allow_duplicates=False)
# Regressor
knn_testX_trans = testX.iloc[:,ids_knn]
knn_predict = neigh.predict(knn_testX_trans)

submission_id =  testDB.Id
submission_pred = knn_predict
sub = pd.DataFrame({'Id':submission_id[:],'median_house_value':submission_pred[:]})
sub.to_csv('submission.csv', index = False)