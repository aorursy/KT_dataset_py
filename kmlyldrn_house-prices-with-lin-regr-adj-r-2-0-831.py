import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import scipy.stats as sts

import statsmodels.api as sm

import math

import warnings



from matplotlib.mlab import PCA as mlabPCA

from statsmodels.tools.eval_measures import mse, rmse

from statsmodels.tsa.stattools import acf



from sklearn import linear_model

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.preprocessing import normalize, scale, StandardScaler, MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.metrics import mean_absolute_error



from scipy.stats.mstats import winsorize

from scipy.stats import zscore, jarque_bera, normaltest, bartlett, levene

from sqlalchemy import create_engine



warnings.filterwarnings('ignore')



import folium
hs=pd.read_csv("../input/kc_house_data.csv")
hs.head()
hs.info()
hs.describe()
house=hs

house["year_sold"]=np.int64([i[0:4] for i in hs["date"]])

house["month_sold"]=np.int64([i[4:6] for i in hs["date"]])

house["day_sold"]=np.int64([i[6:8] for i in hs["date"]])



sq_conv=10.7639

sqft=["sqft_living", "sqft_lot", "sqft_above", "sqft_basement", "sqft_living15", "sqft_lot15"]

sqmt=["sqmt"+i[4:] for i in sqft]

for i in range(0, len(sqft)):

    house[sqmt[i]]=[j/sq_conv for j in hs[sqft[i]]]
house.drop(sqft, axis=1, inplace=True)

house.drop("date", axis=1, inplace=True)
plt.figure(figsize=(20,10))

plt.subplots_adjust(hspace=1, wspace=0.2)

for i in range(1,len(house.columns)):

    plt.subplot(5,5,i)

    sns.boxplot(house.iloc[:,i])

plt.show()
house=house.drop(house[house["bedrooms"]>15].index, axis=0)

house=house.drop(house[house["bedrooms"]==0].index, axis=0)

house=house.drop(house[house["bathrooms"]==0].index, axis=0)

house=house.reset_index(drop=True)



house["bathrooms"]=[math.trunc(i) for i in house["bathrooms"]]

house["yr_renovated"]=house["yr_renovated"].replace(0, house["yr_built"])
house["age_sold"]=house["year_sold"]-house["yr_built"]
prmt_to_soften=["sqmt_living", "sqmt_lot", "sqmt_above", "sqmt_basement", "sqmt_living15", "sqmt_lot15"]

for i in prmt_to_soften:

    house[i]+=1

    house[i+"_log"]=np.log(house[i])
plt.figure(figsize=(20, 35))

plt.tight_layout()



for i in range(1, len(house.columns)-1):

    plt.subplot(8,4,i)

    plt.hist(house[house.columns[i]])

    plt.title(house.columns[i])

    

plt.show()
plt.figure(figsize=(20, 50))

plt.tight_layout()



for i in range(2, len(house.columns)-1):

    plt.subplot(16,4,2*i-3)

    plt.scatter(house[house.columns[i]], house["price"])

    plt.title(house.columns[i]+" (price)")

    

    plt.subplot(16,4,2*i-2)

    plt.scatter(house[house.columns[i]], np.log(house["price"]))

    plt.title(house.columns[i]+" (price is logaritmic)")



plt.subplots_adjust(hspace=0.3, wspace=0.3)

plt.show()
rt=MinMaxScaler()

ratio=rt.fit_transform(house[["price"]])



m = folium.Map(location=[47.55, -122.21], tiles="Mapbox Bright", zoom_start=10)

 

for i in range(0,len(house["long"])):

    if ratio[i]<0.2:

       folium.Circle(

           location=[house["lat"][i], house["long"][i]],

           radius=1,

           fill=True,

           color='yellow',

           fill_color='yellow'

       ).add_to(m)

    

for i in range(0,len(house["long"])):    

    if ratio[i]>=0.2 and ratio[i]<0.4:

       folium.Circle(

           location=[house["lat"][i], house["long"][i]],

           radius=3,

           fill=True,

           color='blue',

           fill_color='blue'

       ).add_to(m)

    

for i in range(0,len(house["long"])):    

    if ratio[i]>=0.4 and ratio[i]<0.6:

       folium.Circle(

           location=[house["lat"][i], house["long"][i]],

           radius=20,

           fill=True,

           color='red',

           fill_color='red'

       ).add_to(m)

    

for i in range(0,len(house["long"])):

    if ratio[i]>=0.6 and ratio[i]<0.8:

       folium.Circle(

           location=[house["lat"][i], house["long"][i]],

           radius=20,

           fill=True,

           color='cyan',

           fill_color='cyan'

       ).add_to(m)

    

for i in range(0,len(house["long"])):

    if ratio[i]>=0.8:

       folium.Circle(

           location=[house["lat"][i], house["long"][i]],

           radius=20,

           fill=True,

           color='black',

           fill_color='black'

       ).add_to(m)



m.save('mymap.html')
house["price_log"]=np.log(house["price"])

house_corr=house.corr()



plt.figure(figsize=(20,15))

sns.heatmap(house_corr, vmin=-1, vmax=1, cmap="bwr", annot=True, linewidth=0.1)

plt.title("Parameter Correlation Matrix")

plt.show()
selected_columns=["price", "bedrooms", "bathrooms","floors", "waterfront", "view", "grade", "lat", 

                  "sqmt_living", "sqmt_above", "sqmt_basement", "sqmt_living15", "price_log"]



reduced=house_corr.loc[selected_columns, selected_columns]



plt.figure(figsize=(12,6))

sns.heatmap(reduced, vmin=-1, vmax=1, cmap="bwr", annot=True, linewidth=0.1)

plt.title("Reduced Parameter Correlation Matrix")

plt.show()
Y = house["price_log"]

X = house[["bedrooms", "bathrooms","floors", "waterfront", "view", "grade", "lat", 

                  "sqmt_living", "sqmt_above", "sqmt_basement", "sqmt_living15"]]



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 465)



print("Observation Quantity in Training Set : {}".format(X_train.shape[0]))

print("Observation Quantity in Test Set     : {}".format(X_test.shape[0]))
# Having less correlation with eachother (also with target) having unique features

model_1=house[["bedrooms", "floors", "waterfront", "view", "lat", "sqmt_basement"]]



# Having less correlation with eachother including 1 high target-correlated parameter

model_2=house[["bedrooms", "floors", "waterfront", "view", "lat", "sqmt_basement", "sqmt_living"]]



# Having less correlation with eachother including 1 (different) high target-correlated parameter

model_3=house[["bedrooms", "floors", "waterfront", "view", "lat", "sqmt_basement", "grade"]]



# Highest target-correlated parameters (May contain heteroskedasticity)

model_4=house[["bathrooms", "grade", "sqmt_living", "sqmt_above", "sqmt_living15"]]



# Balanced (half is less correlated having unique features, and other half is high target-correlated)

model_5=house[["waterfront", "view", "lat", "grade", "sqmt_living", "bathrooms"]]



# 7 components of PCA result

house_stdized = StandardScaler().fit_transform(house)

pca = PCA(n_components=7)

h_std=pca.fit_transform(house_stdized)



pca_result=pd.DataFrame()

for i in range(7):

    pca_result[i]=h_std.T[i]



model_6=pca_result
# MODEL 1

m_1_train, m_1_test, y_train, y_test = train_test_split(model_1, Y, test_size = 0.25, random_state = 465)

m_1_train = sm.add_constant(m_1_train)

result_1 = sm.OLS(y_train, m_1_train).fit()

print(result_1.summary())



# MODEL 2

m_2_train, m_2_test, y_train, y_test = train_test_split(model_2, Y, test_size = 0.25, random_state = 465)

m_2_train = sm.add_constant(m_2_train)

result_2 = sm.OLS(y_train, m_2_train).fit()

print(result_2.summary())



# MODEL 3

m_3_train, m_3_test, y_train, y_test = train_test_split(model_3, Y, test_size = 0.25, random_state = 465)

m_3_train = sm.add_constant(m_3_train)

result_3 = sm.OLS(y_train, m_3_train).fit()

print(result_3.summary())



# MODEL 4

m_4_train, m_4_test, y_train, y_test = train_test_split(model_4, Y, test_size = 0.25, random_state = 465)

m_4_train = sm.add_constant(m_4_train)

result_4 = sm.OLS(y_train, m_4_train).fit()

print(result_4.summary())



# MODEL 5

m_5_train, m_5_test, y_train, y_test = train_test_split(model_5, Y, test_size = 0.25, random_state = 465)

m_5_train = sm.add_constant(m_5_train)

result_5 = sm.OLS(y_train, m_5_train).fit()

print(result_5.summary())



# MODEL 6

m_6_train, m_6_test, y_train, y_test = train_test_split(model_6, Y, test_size = 0.25, random_state = 465)

m_6_train = sm.add_constant(m_6_train)

result_6 = sm.OLS(y_train, m_6_train).fit()

print(result_6.summary())
# Model 1 predicts

m_1_test=sm.add_constant(m_1_test)

result_1_predicted=result_1.predict(m_1_test)

train_1_predicted=result_1.predict(m_1_train)



# Model 2 predicts

m_2_test=sm.add_constant(m_2_test)

result_2_predicted=result_2.predict(m_2_test)

train_2_predicted=result_2.predict(m_2_train)



# Model 3 predicts

m_3_test=sm.add_constant(m_3_test)

result_3_predicted=result_3.predict(m_3_test)

train_3_predicted=result_3.predict(m_3_train)



# Model 4 predicts

m_4_test=sm.add_constant(m_4_test)

result_4_predicted=result_4.predict(m_4_test)

train_4_predicted=result_4.predict(m_4_train)



# Model 5 predicts

m_5_test=sm.add_constant(m_5_test)

result_5_predicted=result_5.predict(m_5_test)

train_5_predicted=result_5.predict(m_5_train)



# Model 6 predicts

m_6_test=sm.add_constant(m_6_test)

result_6_predicted=result_6.predict(m_6_test)

train_6_predicted=result_6.predict(m_6_train)
print("---------- MODEL 1 ----------")

print("Mean Absolute Error (MAE)             : {}".format(mean_absolute_error(y_test, result_1_predicted)))

print("Mean Squared Error (MSE)              : {}".format(mse(y_test, result_1_predicted)))

print("Root Mean Square Error (RMSE)         : {}".format(rmse(y_test, result_1_predicted)))

print("Mean Absolute Percentage Error (MAPE) : {}".format(np.mean(np.abs((y_test - result_1_predicted) / y_test)) * 100))

print("\n")



print("---------- MODEL 2 ----------")

print("Mean Absolute Error (MAE)             : {}".format(mean_absolute_error(y_test, result_2_predicted)))

print("Mean Squared Error (MSE)              : {}".format(mse(y_test, result_2_predicted)))

print("Root Mean Square Error (RMSE)         : {}".format(rmse(y_test, result_2_predicted)))

print("Mean Absolute Percentage Error (MAPE) : {}".format(np.mean(np.abs((y_test - result_2_predicted) / y_test)) * 100))

print("\n")



print("---------- MODEL 3 ----------")

print("Mean Absolute Error (MAE)             : {}".format(mean_absolute_error(y_test, result_3_predicted)))

print("Mean Squared Error (MSE)              : {}".format(mse(y_test, result_3_predicted)))

print("Root Mean Square Error (RMSE)         : {}".format(rmse(y_test, result_3_predicted)))

print("Mean Absolute Percentage Error (MAPE) : {}".format(np.mean(np.abs((y_test - result_3_predicted) / y_test)) * 100))

print("\n")



print("---------- MODEL 4 ----------")

print("Mean Absolute Error (MAE)             : {}".format(mean_absolute_error(y_test, result_4_predicted)))

print("Mean Squared Error (MSE)              : {}".format(mse(y_test, result_4_predicted)))

print("Root Mean Square Error (RMSE)         : {}".format(rmse(y_test, result_4_predicted)))

print("Mean Absolute Percentage Error (MAPE) : {}".format(np.mean(np.abs((y_test - result_4_predicted) / y_test)) * 100))

print("\n")



print("---------- MODEL 5 ----------")

print("Mean Absolute Error (MAE)             : {}".format(mean_absolute_error(y_test, result_5_predicted)))

print("Mean Squared Error (MSE)              : {}".format(mse(y_test, result_5_predicted)))

print("Root Mean Square Error (RMSE)         : {}".format(rmse(y_test, result_5_predicted)))

print("Mean Absolute Percentage Error (MAPE) : {}".format(np.mean(np.abs((y_test - result_5_predicted) / y_test)) * 100))

print("\n")



print("---------- MODEL 6 ----------")

print("Mean Absolute Error (MAE)             : {}".format(mean_absolute_error(y_test, result_6_predicted)))

print("Mean Squared Error (MSE)              : {}".format(mse(y_test, result_6_predicted)))

print("Root Mean Square Error (RMSE)         : {}".format(rmse(y_test, result_6_predicted)))

print("Mean Absolute Percentage Error (MAPE) : {}".format(np.mean(np.abs((y_test - result_6_predicted) / y_test)) * 100))

print("\n")
header_font = {'family':'arial', 'color':'darkred', 'weight':'bold', 'size':15}

axis_font = {'family':'arial', 'color':'darkblue', 'weight':'bold', 'size':12}



plt.figure(figsize=(15,8))



plt.subplot(2,3,1)

plt.scatter(y_train, train_1_predicted, color="brown", label="Training Set")

plt.scatter(y_test, result_1_predicted, color="blue", label="Test Set", alpha=0.2)

plt.plot(y_test, y_test, color="red", label="Price (Logaritmic)")

plt.xlabel("True Prices", fontdict=axis_font)

plt.ylabel("Predicted Prices", fontdict=axis_font)

plt.title("True and Predicted Prices (Model 1)", fontdict=header_font)

plt.legend(loc = "upper left")



plt.subplot(2,3,2)

plt.scatter(y_train, train_2_predicted, color="brown", label="Training Set")

plt.scatter(y_test, result_2_predicted, color="blue", label="Test Set", alpha=0.2)

plt.plot(y_test, y_test, color="red", label="Price (Logaritmic)")

plt.xlabel("True Prices", fontdict=axis_font)

plt.ylabel("Predicted Prices", fontdict=axis_font)

plt.title("True and Predicted Prices (Model 2)", fontdict=header_font)

plt.legend(loc = "upper left")



plt.subplot(2,3,3)

plt.scatter(y_train, train_3_predicted, color="brown", label="Training Set")

plt.scatter(y_test, result_3_predicted, color="blue", label="Test Set", alpha=0.2)

plt.plot(y_test, y_test, color="red", label="Price (Logaritmic)")

plt.xlabel("True Prices", fontdict=axis_font)

plt.ylabel("Predicted Prices", fontdict=axis_font)

plt.title("True and Predicted Prices (Model 3)", fontdict=header_font)

plt.legend(loc = "upper left")



plt.subplot(2,3,4)

plt.scatter(y_train, train_4_predicted, color="brown", label="Training Set")

plt.scatter(y_test, result_4_predicted, color="blue", label="Test Set", alpha=0.2)

plt.plot(y_test, y_test, color="red", label="Price (Logaritmic)")

plt.xlabel("True Prices", fontdict=axis_font)

plt.ylabel("Predicted Prices", fontdict=axis_font)

plt.title("True and Predicted Prices (Model 4)", fontdict=header_font)

plt.legend(loc = "upper left")



plt.subplot(2,3,5)

plt.scatter(y_train, train_5_predicted, color="brown", label="Training Set")

plt.scatter(y_test, result_5_predicted, color="blue", label="Test Set", alpha=0.2)

plt.plot(y_test, y_test, color="red", label="Price (Logaritmic)")

plt.xlabel("True Prices", fontdict=axis_font)

plt.ylabel("Predicted Prices", fontdict=axis_font)

plt.title("True and Predicted Prices (Model 5)", fontdict=header_font)

plt.legend(loc = "upper left")



plt.subplot(2,3,6)

plt.scatter(y_train, train_6_predicted, color="brown", label="Training Set")

plt.scatter(y_test, result_6_predicted, color="blue", label="Test Set", alpha=0.2)

plt.plot(y_test, y_test, color="red", label="Price (Logaritmic)")

plt.xlabel("True Prices", fontdict=axis_font)

plt.ylabel("Predicted Prices", fontdict=axis_font)

plt.title("True and Predicted Prices (Model 6)", fontdict=header_font)

plt.legend(loc = "upper left")



plt.tight_layout()

plt.show()
errors=math.e**result_6_predicted-math.e**y_test

acf_data = acf(errors)



plt.figure(figsize=(15,4))

plt.subplot(1,2,1)

plt.hist(errors/math.e**y_test, bins=100)

plt.title("Distribution of Errors")



plt.subplot(1,2,2)

plt.plot(acf_data[1:])

plt.title("Autocorrelation Grapic")



plt.show()
bart_stats = bartlett(math.e**result_6_predicted, errors)

lev_stats = levene(math.e**result_6_predicted, errors)



print("Bartlett test value : {0:3g} and p value : {1:.21f}".format(bart_stats[0], bart_stats[1]))

print("Levene test value   : {0:3g} and p value : {1:.21f}".format(lev_stats[0], lev_stats[1]))