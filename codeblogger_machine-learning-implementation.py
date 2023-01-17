import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv') #read to file

data
data["class"] = [1 if each == "Hernia" else 0 for each in data["class"]]

# Hernia = 1

# Normal = 0
data
data.info()
df.describe().T
plt.scatter(data.pelvic_radius,data.sacral_slope)

plt.xlabel("pelvic radius")

plt.ylabel("sacral slope")

plt.show()
from sklearn.linear_model import LinearRegression



linear_reg = LinearRegression()



# sacral_slope VS pelvic_radius

print("sacral_slope type: ", type(data.sacral_slope))

print("pelvic_radius type: ", type(data.pelvic_radius))
x = data.sacral_slope.values.reshape(-1,1)

y = data.pelvic_radius.values.reshape(-1,1)



linear_reg.fit(x,y)
b0 = linear_reg.predict([[0]]) # You can write the desired value instead of 0. Here we wrote 0 to find the point where the line crosses the y-axis

print("b0: ", b0)
# another way: 

b0 = linear_reg.intercept_

print("b0: ", b0)
b1 = linear_reg.coef_

print("b1 = ", b1)
print(linear_reg.predict([[45]]))
array = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]).reshape(-1,1)

plt.scatter(x,y)

y_head = linear_reg.predict(array)

plt.plot(array,y_head,color="red")

plt.show()
y = data["class"].values.reshape(-1,1)

x = data.drop(["class"],axis=1).values # axis = 1 => for columns
multiple_linear_regression = LinearRegression()

multiple_linear_regression.fit(x,y)
print("b0: ",multiple_linear_regression.intercept_)  # The point where it intersects the y axis

print("b1,b2,b3,b4,b5,b6: ",multiple_linear_regression.coef_) # The slopes of the line
multiple_linear_regression.predict(np.array([[

    57.26,

    19.98,

    38.63,

    31.43,

    115.098,

    4.4512

]]))
x = data["class"].values.reshape(-1,1)

y = data["sacral_slope"].values.reshape(-1,1)
from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor()

tree_reg.fit(x,y)
y_head = tree_reg.predict(x)



plt.scatter(x,y,color="red")

plt.plot(x,y_head,color="green")

plt.xlabel("Grandstand Level")

plt.ylabel("Price")

plt.show()
plt.scatter(data[data["class"]==1].pelvic_radius,data[data["class"]==1].sacral_slope,color="red",label="hernia",alpha= 0.6)

plt.scatter(data[data["class"]==0].pelvic_radius,data[data["class"]==0].sacral_slope,color="green",label="normal",alpha= 0.6)

plt.xlabel("pelvic_radius")

plt.ylabel("sacral_slope")

plt.legend()

plt.show()
y = data["class"].values

x_data = data.drop(["class"],axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))
# train test split

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)
# knn model

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=3) # n_neighbors => key count

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)
prediction # looks nice :)
print("{} knn score: {} ".format(3,knn.score(x_test,y_test)*100)) # accuracy = 79.5%
# find k value

score_list = []



for each in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,15),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
print("{} knn score: {} ".format(13,knn.score(x_test,y_test)*100))
from IPython.display import Image

Image(url="https://www.researchgate.net/publication/304611323/figure/fig8/AS:668377215406089@1536364954428/Classification-of-data-by-support-vector-machine-SVM.png")
y = data["class"].values

x_data = data.drop(["class"],axis=1)



# normalazition

x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))



# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.svm import SVC



svm = SVC(random_state=42)

svm.fit(x_train,y_train)



print("Accuracy of SVM algo: ", svm.score(x_test,y_test)*100)
y = data["class"].values

x_data = data.drop(["class"],axis=1)



# normalazition

x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))



# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.naive_bayes import GaussianNB



nb = GaussianNB()

nb.fit(x_train,y_train)

print("Accuracy of naive_bayes algo:",nb.score(x_test,y_test)*100)
y = data["class"].values

x_data = data.drop(["class"],axis=1)



# normalazition

x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))



# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier(random_state=42)

dt.fit(x_train,y_train)



print("Score: ", dt.score(x_test,y_test)*100)
y = data["class"].values

x_data = data.drop(["class"],axis=1)



# normalazition

x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))



# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=100,random_state=42) # n_estimators = number of trees

rf.fit(x_train,y_train)

print("Score: ", rf.score(x_test,y_test)*100)
y = data["class"].values.reshape(-1,1)

x_data = data.drop(["class"],axis=1).values



# normalazition

x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))



linear_reg.fit(x,y)

y_head = linear_reg.predict(x) 
from sklearn.metrics import r2_score



print("r_square score: ",r2_score(y,y_head))