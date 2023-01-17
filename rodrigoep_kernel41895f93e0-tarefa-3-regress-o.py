import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import os
train = pd.read_csv("../input/cahouse/train.csv", encoding ='utf-8')
test = pd.read_csv("../input/cahouse/test.csv", encoding ='utf-8')
train.head()
train.info()
train["median_rooms"] = train["total_rooms"]/train["households"]        
train["median_people"] = train["total_bedrooms"]/train["population"]
train["median_bedrooms"] = train["total_bedrooms"]/train["households"]

test["median_rooms"] = test["total_rooms"]/test["households"]
test["median_people"] = test["total_bedrooms"]/test["population"]
test["median_bedrooms"] = test["total_bedrooms"]/test["households"]
train = train[train["median_house_value"] <= 500000]
places = ["Santa Monica State CA", "Silicon Valley CA", "Los Angeles CA", "San Francisco CA", "Beverly Hills"]
from geopy.geocoders import Nominatim
for i in range(len(places)):   
    geolocator = Nominatim(user_agent="specify_your_app_name_here")
    location = geolocator.geocode(places[i])
    s = places[i]
    train[s] = np.sqrt((location.latitude - train["latitude"])**2 + (location.longitude - train["longitude"])**2)
    test[s] =  np.sqrt((location.latitude - test["latitude"])**2 + (location.longitude - test["longitude"])**2)
train.pop("Id")
Id = test.pop("Id")
def correlation_matrix(df):
    
    from matplotlib import cm as cm
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Text Feature Correlation')
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.rcParams['figure.figsize'] = [20,5]
    plt.show()

correlation_matrix(train)
Target = train.pop("median_house_value")
train.shape
test.shape
train.hist(figsize=(12,8),bins=50)
Target.hist(figsize=(6,4),bins=50)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
train = sc.fit_transform(train)  
test = sc.transform(test)

from sklearn.decomposition import PCA
pca = PCA()  
train = pca.fit_transform(train)  
test = pca.transform(test) 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression, Lasso, Ridge
Lr = Lasso()
scores = cross_val_score(Lr, train, Target, cv=10, scoring = "neg_mean_squared_error")
print("Erro RSME médio: ", np.sqrt(-scores.mean())/Target.mean())
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors = 10)
scores = cross_val_score(knn, train, Target, cv=10, scoring = "neg_mean_squared_error")
print("Erro RSME médio: ", np.sqrt(-scores.mean())/Target.mean())
from sklearn.ensemble import RandomForestRegressor
Rfr = RandomForestRegressor(n_estimators = 20)
scores = cross_val_score(Rfr, train, Target, cv = 10, scoring = "neg_mean_squared_error")
print("Erro RSME médio: ", np.sqrt(-scores.mean())/Target.mean())
Rfr.fit(train,Target)
pred = Rfr.predict(test)
for i in range(len(pred)):
    if pred[i] < 0:
        pred[i] = -1*pred[i]
output =pd.DataFrame(Id)
output["median_house_value"] = pred
output.to_csv("tarefa3.aa", index = False)