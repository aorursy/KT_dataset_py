#Imports

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from math import *

from scipy.spatial import distance



#Load the data and read all the columns

data = pd.read_csv("../input/parkinsons/parkinsons.csv")

print("Data : parkinsons.csv \n ")

print(data) 
#Display data

X = data["HNR"]

Y = data["NHR"]

plt.plot(X, Y, marker="o")

plt.xlabel('HNR')

plt.ylabel('NHR')

plt.title("HNR and NHR data")

plt.show()

#mean of NHR

print("Mean of NHR : ")

print(data["NHR"].mean())



#mean of HNR

print("Mean of HNR : ")

print(data["HNR"].mean())



#data point P

p = (data["HNR"].mean(),data["NHR"].mean())

print("Data Point P : ")

print(p)



#Euclidean distance

cols = ["HNR","NHR"]

Euclidean = np.linalg.norm(p - data[cols].values,axis=1)

data['Euclidean'] = Euclidean



#Manhattan block metric

Manhattan = (abs(p[0]-data["HNR"].values) + abs(p[1]-data["NHR"].values))

data['Manhattan'] = Manhattan



#Minkowski metric

a1 = (abs(p[0]-data["HNR"].values))

a2 = (abs(p[1]-data["NHR"].values))

r1 = (np.power(a1,6)+ np.power(a2,6))

Minkowski = np.power(r1, 1/6)

data['Minkowski'] = Minkowski



#Chebyshev distance

c1 = abs(p[0]-data["HNR"].values) 

c2 = abs(p[1]-data["NHR"].values)

Chebyshev = np.maximum(c1, c2)

data['Chebyshev'] = Chebyshev



#Cosine distance.

cosine = data.apply(lambda x: distance.cosine([x.HNR,p[0]],[x.NHR,p[1]]),axis=1)

data['Cosine'] = cosine





print(data[['HNR','NHR','Euclidean','Manhattan','Minkowski','Chebyshev','Cosine']])
#Five closest points for Euclidean Distance

e_data = data.copy()

e_data.sort_values(by=['Euclidean'], ascending=True,inplace=True)

print("Five Nearest Points using Euclidean Distance : ")

print(e_data[['HNR','NHR','Euclidean']].head(5))



plt.plot([p[0],e_data["HNR"][78]],[p[1],e_data["NHR"][78]],marker="o", markerfacecolor="b")

plt.plot([p[0],e_data["HNR"][114]],[p[1],e_data["NHR"][114]],marker="o", markerfacecolor="b")

plt.plot([p[0],e_data["HNR"][9]],[p[1],e_data["NHR"][9]],marker="o", markerfacecolor="b")

plt.plot([p[0],e_data["HNR"][109]],[p[1],e_data["NHR"][109]],marker="o", markerfacecolor="b")

plt.plot([p[0],e_data["HNR"][154]],[p[1],e_data["NHR"][154]],marker="o", markerfacecolor="b")

plt.plot([p[0],e_data["HNR"][8]],[p[1],e_data["NHR"][8]],marker="o", markerfacecolor="b")

plt.plot([p[0],e_data["HNR"][69]],[p[1],e_data["NHR"][69]],marker="o", markerfacecolor="b")

plt.plot([p[0],e_data["HNR"][25]],[p[1],e_data["NHR"][25]],marker="o", markerfacecolor="b")

plt.plot([p[0],e_data["HNR"][95]],[p[1],e_data["NHR"][95]],marker="o", markerfacecolor="b")

plt.plot([p[0],e_data["HNR"][184]],[p[1],e_data["NHR"][184]],marker="o", markerfacecolor="b")

plt.plot(p[0],p[1],marker="o", markerfacecolor="r")

plt.title('Euclidean Distance')

plt.xlabel('HNR')

plt.ylabel('NHR')



plt.show()

#Five closest points for Manhattan Distance

m_data = data.copy()

m_data.sort_values(by=['Manhattan'], ascending=True,inplace=True)

print("\n Five Nearest Points using Manhattan Distance : ")

print(m_data[['HNR','NHR','Manhattan']].head(5))



plt.plot([p[0],m_data["HNR"][78]],[p[1],m_data["NHR"][78]],marker="o", markerfacecolor="b")

plt.plot([p[0],m_data["HNR"][114]],[p[1],m_data["NHR"][114]],marker="o", markerfacecolor="b")

plt.plot([p[0],m_data["HNR"][9]],[p[1],m_data["NHR"][9]],marker="o", markerfacecolor="b")

plt.plot([p[0],m_data["HNR"][109]],[p[1],m_data["NHR"][109]],marker="o", markerfacecolor="b")

plt.plot([p[0],m_data["HNR"][154]],[p[1],m_data["NHR"][154]],marker="o", markerfacecolor="b")

plt.plot([p[0],m_data["HNR"][8]],[p[1],m_data["NHR"][8]],marker="o", markerfacecolor="b")

plt.plot([p[0],m_data["HNR"][69]],[p[1],m_data["NHR"][69]],marker="o", markerfacecolor="b")

plt.plot([p[0],m_data["HNR"][25]],[p[1],m_data["NHR"][25]],marker="o", markerfacecolor="b")

plt.plot([p[0],m_data["HNR"][95]],[p[1],m_data["NHR"][95]],marker="o", markerfacecolor="b")

plt.plot([p[0],m_data["HNR"][184]],[p[1],m_data["NHR"][184]],marker="o", markerfacecolor="b")

plt.plot(p[0],p[1],marker="o", markerfacecolor="r")

plt.title('Manhattan Distance')

plt.xlabel('HNR')

plt.ylabel('NHR')

plt.show()
#Five closest points for Minkowski Distance

mi_data = data.copy()

mi_data.sort_values(by=['Minkowski'], ascending=True,inplace=True)

print("\n Five Nearest Points using Minkowski Distance : ")

print(mi_data[['HNR','NHR','Minkowski']].head(5))



plt.plot([p[0],mi_data["HNR"][78]],[p[1],mi_data["NHR"][78]],marker="o", markerfacecolor="b")

plt.plot([p[0],mi_data["HNR"][114]],[p[1],mi_data["NHR"][114]],marker="o", markerfacecolor="b")

plt.plot([p[0],mi_data["HNR"][9]],[p[1],mi_data["NHR"][9]],marker="o", markerfacecolor="b")

plt.plot([p[0],mi_data["HNR"][109]],[p[1],mi_data["NHR"][109]],marker="o", markerfacecolor="b")

plt.plot([p[0],mi_data["HNR"][154]],[p[1],mi_data["NHR"][154]],marker="o", markerfacecolor="b")

plt.plot([p[0],mi_data["HNR"][8]],[p[1],mi_data["NHR"][8]],marker="o", markerfacecolor="b")

plt.plot([p[0],mi_data["HNR"][69]],[p[1],mi_data["NHR"][69]],marker="o", markerfacecolor="b")

plt.plot([p[0],mi_data["HNR"][25]],[p[1],mi_data["NHR"][25]],marker="o", markerfacecolor="b")

plt.plot([p[0],mi_data["HNR"][95]],[p[1],mi_data["NHR"][95]],marker="o", markerfacecolor="b")

plt.plot([p[0],mi_data["HNR"][184]],[p[1],mi_data["NHR"][184]],marker="o", markerfacecolor="b")

plt.plot(p[0],p[1],marker="o", markerfacecolor="r")

plt.title('Minkowski Distance')

plt.xlabel('HNR')

plt.ylabel('NHR')

plt.show()

#Five closest points for Chebyshev Distance

c_data = data.copy()

c_data.sort_values(by=['Chebyshev'], ascending=True,inplace=True)

print("\n Five Nearest Points using Chebyshev Distance : ")

print(c_data[['HNR','NHR','Chebyshev']].head(5))



plt.plot([p[0],c_data["HNR"][78]],[p[1],c_data["NHR"][78]],marker="o", markerfacecolor="b")

plt.plot([p[0],c_data["HNR"][114]],[p[1],c_data["NHR"][114]],marker="o", markerfacecolor="b")

plt.plot([p[0],c_data["HNR"][9]],[p[1],c_data["NHR"][9]],marker="o", markerfacecolor="b")

plt.plot([p[0],c_data["HNR"][109]],[p[1],c_data["NHR"][109]],marker="o", markerfacecolor="b")

plt.plot([p[0],c_data["HNR"][154]],[p[1],c_data["NHR"][154]],marker="o", markerfacecolor="b")

plt.plot([p[0],c_data["HNR"][8]],[p[1],c_data["NHR"][8]],marker="o", markerfacecolor="b")

plt.plot([p[0],c_data["HNR"][69]],[p[1],c_data["NHR"][69]],marker="o", markerfacecolor="b")

plt.plot([p[0],c_data["HNR"][25]],[p[1],c_data["NHR"][25]],marker="o", markerfacecolor="b")

plt.plot([p[0],c_data["HNR"][95]],[p[1],c_data["NHR"][95]],marker="o", markerfacecolor="b")

plt.plot([p[0],c_data["HNR"][184]],[p[1],c_data["NHR"][184]],marker="o", markerfacecolor="b")

plt.plot(p[0],p[1],marker="o", markerfacecolor="r")

plt.title('Chebyshev Distance')

plt.xlabel('HNR')

plt.ylabel('NHR')

plt.show()

#Five closest points for Cosine Distance

co_data = data.copy()

co_data.sort_values(by=['Cosine'], ascending=True,inplace=True)

print("\n Five Nearest Points using Cosine Distance : ")

print(co_data[['HNR','NHR','Cosine']].head(5))



plt.plot([p[0],co_data["HNR"][69]],[p[1],co_data["NHR"][69]],marker="o", markerfacecolor="b")

plt.plot([p[0],co_data["HNR"][66]],[p[1],co_data["NHR"][66]],marker="o", markerfacecolor="b")

plt.plot([p[0],co_data["HNR"][79]],[p[1],co_data["NHR"][79]],marker="o", markerfacecolor="b")

plt.plot([p[0],co_data["HNR"][155]],[p[1],co_data["NHR"][155]],marker="o", markerfacecolor="b")

plt.plot([p[0],co_data["HNR"][143]],[p[1],co_data["NHR"][143]],marker="o", markerfacecolor="b")

plt.plot([p[0],co_data["HNR"][122]],[p[1],co_data["NHR"][122]],marker="o", markerfacecolor="b")

plt.plot([p[0],co_data["HNR"][136]],[p[1],co_data["NHR"][136]],marker="o", markerfacecolor="b")

plt.plot([p[0],co_data["HNR"][124]],[p[1],co_data["NHR"][124]],marker="o", markerfacecolor="b")

plt.plot([p[0],co_data["HNR"][0]],[p[1],co_data["NHR"][0]],marker="o", markerfacecolor="b")

plt.plot([p[0],co_data["HNR"][92]],[p[1],co_data["NHR"][92]],marker="o", markerfacecolor="b")

plt.plot(p[0],p[1],marker="o", markerfacecolor="r")

plt.title('Cosine Distance')

plt.xlabel('HNR')

plt.ylabel('NHR')

plt.show()