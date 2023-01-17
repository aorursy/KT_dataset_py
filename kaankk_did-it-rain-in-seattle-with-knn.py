import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier



data = pd.read_csv("../input/did-it-rain-in-seattle-19482017/seattleWeather_1948-2017.csv")

data =data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]

print(data.head())
data.drop(["DATE"],axis = 1,inplace = True)



print(data.head())
t = data[data.RAIN == True]

f = data[data.RAIN == False]



plt.figure(1)

plt.scatter(t.TMAX,t.TMIN,color="red",label="TRUE",alpha= 0.3)

plt.scatter(f.TMAX,f.TMIN,color="green",label="FALSE",alpha= 0.3)



plt.xlabel("TMAX")

plt.ylabel("TMIN")

plt.legend()

plt.show()
data.RAIN = [1 if each == True else 0 for each in data.RAIN]

y = data.RAIN.values

x_data = data.drop(["RAIN"], axis = 1)
x = (x_data -np.min(x_data))/ (np.max(x_data) - np.min(x_data))

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=1)
knn = KNeighborsClassifier(n_neighbors = 2) 

knn.fit(x_train,y_train)



prediction = knn.predict(x_test)
print("{} nn score {}".format(2,knn.score(x_test,y_test)))
score_list=[]

for i in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors = i)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

plt.figure(2)    

plt.plot(range(1,15),score_list)

plt.ylabel(" reliability scores")

plt.xlabel("k values")

plt.show()