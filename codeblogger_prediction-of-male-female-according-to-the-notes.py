import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



data = pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")

data
data = data.drop(["race/ethnicity","parental level of education","lunch","test preparation course"],axis=1) # axis = 1 => for columns

data
# If there is a line with a null value, let's delete it.

data = data.dropna()
data.info()
data.describe().T
# male = 0 | female = 1

print("gender: ", data.gender.value_counts())

data["gender"] = [ 0 if each == "male" else 1 for each in data.gender]

data
plt.scatter(data[data["gender"] == 1]["writing score"],data[data["gender"] == 1]["reading score"],color="blue",label="famale",alpha= 0.3)

plt.scatter(data[data["gender"] == 0]["writing score"],data[data["gender"] == 0]["reading score"],color="red",label="male",alpha= 0.3)

plt.xlabel("radius_mean")

plt.ylabel("texture_mean")

plt.legend()
df = data.copy()
from IPython.display import Image

Image(url="https://i.ibb.co/KL2vG7W/knn2.jpg")
y = data.gender.values

x_data = data.drop(["gender"],axis=1)

x_data
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

x
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)



print("x_train",x_train.shape)

print("x_test",x_test.shape)

print("y_train",y_train.shape)

print("y_test",y_test.shape)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train,y_train)



prediction = knn.predict(x_test)
print("{} nn score: {} ".format(5,knn.score(x_test,y_test)*100))
score_list = []



for each in range(20,40):

    knn2 = KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(20,40),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
knn = KNeighborsClassifier(n_neighbors=30)

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)



print("{} KNN Score: {} ".format(30,knn.score(x_test,y_test)*100))
data = df

data
y = data.gender.values

x_data = data.drop(["gender"],axis=1)



# normalization

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))



# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.svm import SVC



svm = SVC(random_state=42)

svm.fit(x_train,y_train)



print("Accuracy of SVM Algo: ", svm.score(x_test,y_test)*100)
Image(url="https://i.ibb.co/YpP7JY1/1-39-U1-Ln3t-Sd-Fqsf-Qy6ndx-OA.png")
data = df

data
y = data.gender.values

x_data = data.drop(["gender"],axis=1)



# normalization

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))



# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.naive_bayes import GaussianNB



nb = GaussianNB()

nb.fit(x_train,y_train)

print("Accuracy of Naive Bayes Algo:",nb.score(x_test,y_test)*100)