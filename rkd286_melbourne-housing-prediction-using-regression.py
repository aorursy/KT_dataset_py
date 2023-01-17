# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
housing = pd.read_csv('../input/Melbourne_housing_FULL.csv')
housing.head()
housing.info()
plt.figure(figsize=(15,8))
sns.heatmap(housing.isnull(),cbar=False,yticklabels='')
pd.DataFrame(housing.groupby(('Suburb','Rooms'))['Price'].mean()).head(10)
room_avg = housing.groupby('Rooms')['Price'].mean()
room_avg
plt.figure(figsize=(15,8))
sns.boxplot(housing['Rooms'],housing['Price'])
def fill_price(cols):
    price = cols[0]
    rooms = cols[1]
    
    if pd.isnull(price):
        for index,room in room_avg.iteritems():
            if rooms == index:
               return room
    else:
        return price
housing['Price'] = housing[['Price','Rooms']].apply(fill_price, axis=1)
print("Null Values in the Price Column: {}".format(sum(housing['Price'].isnull())))
#Average number of rooms for Houses with less than 3 Rooms
print("Average Number of Bathrooms for houses with < 3 Rooms: {}".format(housing['Bathroom'][housing['Rooms'] < 3].mean()))


#Average number of rooms for Houses with greater than 3 Rooms
print("Average Number of Bathrooms for houses with > 3 Rooms: {}".format(housing['Bathroom'][housing['Rooms'] > 3].mean()))

def fill_bathroom(col):
    bathrooms = col[0]
    rooms = col[1]
    
    if pd.isnull(bathrooms):
        if rooms > 3:
            return 2
        else:
            return 1
    else:
        return bathrooms

housing['Bathroom'] = housing[['Bathroom','Rooms']].apply(fill_bathroom, axis=1)
print("Null Values in the Bathrooom Column: {}".format(sum(housing['Bathroom'].isnull())))
#Average number of carslots for Houses with less than 3 Rooms
print("Average Number of Carslots for houses with < 3 Rooms: {}".format(housing['Car'][housing['Rooms'] < 3].mean()))


#Average number of carslots for Houses with greater than 3 Rooms
print("Average Number of Carslots for houses with > 3 Rooms: {}".format(housing['Car'][housing['Rooms'] > 3].mean()))
def fill_car(col):
    car = col[0]
    rooms = col[1]
    
    if pd.isnull(car):
        if rooms > 3:
            return 2
        else:
            return 1
    else:
        return car

housing['Car'] = housing[['Car','Rooms']].apply(fill_bathroom, axis=1)
print("Null Values in the Car Column: {}".format(sum(housing['Car'].isnull())))
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(housing['Suburb'])
suburb_encoding = le.transform(housing['Suburb'])
suburb_encoding
housing['Suburb Encoding'] = suburb_encoding
final = housing[['Suburb Encoding','Price','Distance','Rooms','Bathroom','Car']]
final.dropna(inplace=True)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
lr = LinearRegression()
X = final.drop('Price',axis=1)
y= final['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
lr.fit(X_train,y_train)
price_pred_lr = lr.predict(X_test)
lr_r2 = metrics.r2_score(y_test,price_pred_lr)
lr_rmse = np.sqrt(metrics.mean_squared_error(y_test,price_pred_lr))
print("R2 Score for Linear Regression: {0:0.2f}".format(lr_r2))
print("RMSE for Linear Regression: {}".format(lr_rmse))
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(X_train,y_train)
price_pred_dtr = dtr.predict(X_test)
dtr_r2 = metrics.r2_score(y_test,price_pred_dtr)
dtr_rmse = np.sqrt(metrics.mean_squared_error(y_test,price_pred_dtr))
print("R2 Score for Decision Tree: {0:0.2f}".format(dtr_r2))
print("RMSE for Decision Tree: {}".format(dtr_rmse))
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=20)
rfr.fit(X_train,y_train)
price_pred_rfr = rfr.predict(X_test)
rfr_r2 = metrics.r2_score(y_test,price_pred_rfr)
rfr_rmse = np.sqrt(metrics.mean_squared_error(y_test,price_pred_rfr))
print("R2 Score for Random Forest: {0:0.2f}".format(rfr_r2))
print("RMSE for Random Forest: {}".format(rfr_rmse))
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train,y_train)
price_pred_knn = knn.predict(X_test)
knn_r2 = metrics.r2_score(y_test,price_pred_knn)
knn_rmse = np.sqrt(metrics.mean_squared_error(y_test,price_pred_knn))
print("R2 Score for KNN: {0:0.2f}".format(knn_r2))
print("RMSE for KNN: {}".format(knn_rmse))
conc_table = pd.DataFrame(data=[[lr_r2,lr_rmse],[dtr_r2,dtr_rmse],[rfr_r2,rfr_rmse],[knn_r2,knn_rmse]], index=['Linear Regression','Decision Trees','Random Forest','K Nearest Neighbor'],columns=['R2 Score','RMSE']).round(2)
conc_table
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
sns.pointplot(conc_table.index, conc_table['R2 Score'],color='red',)
plt.xticks(rotation=70)
plt.xlabel("Model").set_size(20)
plt.ylabel("R2 Score").set_size(20)
plt.title("R2 Score Comparison").set_size(20)
plt.subplot(1,2,2)
sns.pointplot(conc_table.index, conc_table['RMSE'])
plt.xticks(rotation=70)
plt.xlabel("Model").set_size(20)
plt.ylabel("RMSE Score").set_size(20)
plt.title("RMSE Comparison").set_size(20)
plt.tight_layout()