# Import of librairies

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# import data from  data.gouv, csv format

data = pd.read_csv("../input/33.csv")
# First vizualisation

data.head()
data.shape
data.info()
# Delete duplicate

data.drop_duplicates(subset = "valeur_fonciere", inplace = True)

data.shape
# Selection of interest columns in a new dataframe called "dataClean"

dataClean= data[["date_mutation","valeur_fonciere","code_postal","nombre_pieces_principales","surface_terrain","surface_reelle_bati","type_local"]]
# In our data, there is dependancies and industrial local. We delete them because it's very different from house and apartment 

dataClean['type_local'].value_counts()
dataClean = dataClean.drop(dataClean[dataClean["type_local"] == "Dépendance"].index)
dataClean = dataClean.drop(dataClean[dataClean["type_local"] == "Local industriel. commercial ou assimilé"].index)
# Delete of Nan values

dataClean.dropna(inplace=True)
dataClean.info()
sns.distplot(dataClean.valeur_fonciere)
dataClean.valeur_fonciere.describe()
# delete extreme value : 

#house price higher than 4 M€ and lower than 25 K€

# Number of room = 0

# surface lower than 11



dataClean = dataClean[dataClean.valeur_fonciere < 4000000]

dataClean = dataClean[dataClean.valeur_fonciere > 25000]

dataClean = dataClean[dataClean.nombre_pieces_principales > 0]

dataClean = dataClean[dataClean.surface_reelle_bati > 10]



dataClean.shape
# Creation of a dictionnary

local = {"Appartement":1 , "Maison" : 0}
# Replacement of values house by 0 and apartment by 1

dataClean.replace(local, inplace=True)
dataClean.type_local.value_counts()
# Transformation of floats in integer

dataClean = dataClean.astype({"valeur_fonciere" : int , "code_postal" : int, "nombre_pieces_principales":int,"surface_terrain":int,"surface_reelle_bati":int, "type_local" : int})

dataClean.dtypes
# New visialization

dataClean.head(5)
# Creation of the column "agglomeration" with postal code.

dataClean ["agglomeration"] = dataClean["code_postal"]

# Transforming Bordeaux postal code by the value 1. 



dataClean.loc[dataClean["agglomeration"] == 33000, "agglomeration"  ]= 1

dataClean.loc[dataClean["agglomeration"] == 33100, "agglomeration" ] = 1

dataClean.loc[dataClean["agglomeration"] == 33200,"agglomeration" ] = 1

dataClean.loc[dataClean["agglomeration"] == 33300,"agglomeration" ] = 1

dataClean.loc[dataClean["agglomeration"] == 33800,"agglomeration" ] = 1









# Transformation of suburbs postal code by the value 2.





dataClean.loc[dataClean["agglomeration"] == 33130, "agglomeration"] = 2

dataClean.loc[dataClean["agglomeration"] == 33400,"agglomeration" ] = 2

dataClean.loc[dataClean["agglomeration"] == 33600, "agglomeration"] = 2

dataClean.loc[dataClean["agglomeration"] == 33140,"agglomeration" ] = 2

dataClean.loc[dataClean["agglomeration"] == 33700,"agglomeration" ] = 2

dataClean.loc[dataClean["agglomeration"] == 33110,"agglomeration"] = 2

dataClean.loc[dataClean["agglomeration"] == 33520,"agglomeration" ] = 2

dataClean.loc[dataClean["agglomeration"] == 33310, "agglomeration"] = 2

dataClean.loc[dataClean["agglomeration"] == 33150,"agglomeration" ] = 2

dataClean.loc[dataClean["agglomeration"] == 33270,"agglomeration" ] = 2

# Transformation of all others postal code by the value 3.

dataClean.loc[dataClean["agglomeration"] >  2, "agglomeration" ] = 3

# Counting of the agglomeration values

dataClean.agglomeration.value_counts()
# Visualization

dataClean.head()
dataClean["squareMeterPrice"] = round(dataClean.valeur_fonciere / dataClean.surface_reelle_bati).astype(int)

dataClean.head()
# Link between surface and price : the link is not obvious on this graph

plt.figure(figsize=(10,6))

sns.scatterplot(x="surface_reelle_bati",y="valeur_fonciere",data=dataClean).set_title("link between surface and price")



# The square meter price is around 8 000 € in Bordeaux, 4 500 € in the suburbs, and 3 500 € in the rest of Gironde

plt.figure(figsize=(10,6))



sns.barplot(x="agglomeration" , y="squareMeterPrice", data=dataClean).set_title("link between area and price")
# link between type of goods and price : apartement are more expensive than house, probably because they are more in city center

plt.figure(figsize=(10,6))

sns.barplot(x="type_local",y="valeur_fonciere",data=dataClean).set_title("link between house price and type")
# Excepts for the 1 room house (wrong data?), the house are more expensive with more rooms.

plt.figure(figsize=(10,6))



sns.barplot(x="nombre_pieces_principales",y="valeur_fonciere",data=dataClean).set_title("Price and number of room")

plt.show()
# Strong link between the price and the area

plt.figure(figsize=(10,6))

sns.barplot(x="agglomeration",y="valeur_fonciere",data=dataClean).set_title("Link between area and price")

plt.show()
correlation = dataClean.corr()

correlation["valeur_fonciere"].sort_values(ascending=False)
# Splitting between train and test set and selection of x and y

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



x=dataClean[["nombre_pieces_principales", "surface_reelle_bati","type_local", "agglomeration" ]]

y=dataClean["valeur_fonciere"]





X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)



scalerX = StandardScaler().fit(X_train)



X_train = scalerX.transform(X_train)

X_test = scalerX.transform(X_test)

print (X_train.shape)

print (X_test.shape)

print (y_train.shape)

print (y_test.shape)
# KNN

from sklearn import neighbors

knn = neighbors.KNeighborsRegressor(n_neighbors = 44)



# model training

knn.fit(X_train, y_train)



# prediction according the X_test

prediction = knn.predict(X_test)

# RMSE

from sklearn.metrics import mean_squared_error 

from math import sqrt
knnError = sqrt(mean_squared_error(y_test,prediction))

print(knnError)
# best value for k neighbors : 44



rmse_val = [] #to store rmse values for different k

for K in range(50):

    K = K+1

    model = neighbors.KNeighborsRegressor(n_neighbors = K)



    model.fit(X_train, y_train)  #fit the model

    pred=model.predict(X_test) #make prediction on test set

    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse

    rmse_val.append(error) #store rmse values

    print('RMSE value for k= ' , K , 'is:', error)
curve = pd.DataFrame(rmse_val) #elbow curve 

curve.plot()
# R2 score

from sklearn.metrics import r2_score



y_true = y_test

y_pred = prediction

r2_score(y_true, y_pred)  
# Linear Regression



from sklearn.linear_model import LinearRegression



linreg = LinearRegression()



linreg.fit(X_train,y_train)



linregPrediction = linreg.predict(X_test)



linregError = sqrt(mean_squared_error(y_test,linregPrediction))



print(linregError)

print (r2_score(y_test, linregPrediction))
# Bayesian Regression





from sklearn import linear_model

bayesian = linear_model.BayesianRidge()

bayesian.fit(X_train, y_train)



bayesianPredict = bayesian.predict(X_test)

bayesianError = sqrt(mean_squared_error(y_test,bayesianPredict))

print(bayesianError)

print (r2_score(y_test, bayesianPredict))
# Summary : the 3 algorythm are roughtly as good with our data



print ("The RMSE with the linear Regression alorythm is",linregError, "\n")

print ("The RMSE with the Bayesian alorythm is", bayesianError, "\n")

print ("The RMSE with the KNN alorythm is", knnError, "\n")
# Test with 2 cases : a house in the suburb of 130 square meter, and a flat in bordeaux of 40 square meter

suburbHouse = [[7,138,0,2]]

appartmentBordeaux = [[2, 40, 1, 1]]



print(knn.predict(suburbHouse))

print(linreg.predict(suburbHouse))

print(bayesian.predict(suburbHouse),"\n")



print(knn.predict(appartmentBordeaux))

print(linreg.predict(appartmentBordeaux))

print(bayesian.predict(appartmentBordeaux))

