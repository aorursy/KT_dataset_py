

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import sklearn as sns

from sklearn.linear_model import LinearRegression



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df=pd.read_csv("../input/Experiance_and_salary_dataset.csv",sep=";") #read data

df #dataset

df.columns = ["deneyim","maas"] #column rename

df[:5] #data between zero and five
plt.scatter(df.deneyim

            ,df.maas

            ,marker="^")

plt.title("--Linear Regression--")

plt.xlabel("Deneyim")

plt.ylabel("Maaş")

plt.show()
lg=LinearRegression()

x=df.deneyim.values.reshape(-1,1) #Convert numpy array and reshape for linear_regrassion

y=df.maas.values.reshape(-1,1)    #Convert numpy array and reshape for linear_regrassion

lg.fit(x,y) #I creatied a linear_regression model in x and y


plt.scatter(df.deneyim,df.maas,marker="^")

plt.title("--Linear Regression--")

plt.xlabel("Deneyim")

plt.ylabel("Maaş")

plt.show()
b0=lg.predict([[9]]) #sample. How many money salary do I get at the age of 100 ?

b0
b0_kesisim=lg.intercept_ #intersection

b0_kesisim
b1 = lg.coef_  #coefficient

b1
deneyim = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]).reshape(-1,1) #add experiance
plt.scatter(x,y)

plt.title("--Linear Regression--")

plt.xlabel("Deneyim")

plt.ylabel("Maaş")

y_head = lg.predict(deneyim)

plt.plot(deneyim,y_head,color="red")

plt.show()

lg.predict([[100]]) #How many money salary do I get at the age of 100 ?
## New sample 
# The scores of the players according to their age are given in the dataset.

#According to this dataset, the scores of the players can be calculated according to their age.

 
age = pd.DataFrame(np.array([15,20,25,35,40,27,32]) , columns=["yas"])

score = pd.DataFrame(np.array([50,100,125,128,250,135,55]), columns=["puan"])

dataset=pd.concat([age,score],axis=1)
dataset
reg=LinearRegression()

reg.fit(age,score)
reg.predict([[44]])
yas_ = np.array([0,28,30,42,45]).reshape(-1,1) # query for add age range
plt.scatter(dataset.puan,dataset.yas)

plt.title("--Linear Regression--")

plt.xlabel("Puan")

plt.ylabel("Yaş")

y_head = reg.predict(yas_)

plt.plot(y_head,yas_,color="orange")

plt.show();
reg.predict(yas_)