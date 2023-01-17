import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for visualizations

data = pd.read_csv("../input/column_2C_weka.csv")
data.info()
data.head()
data.tail()
data.describe()
data.rename(columns={
    'class': 'symptom_class'
}, inplace=True)
abnormal = data[data.symptom_class == "Abnormal"]
normal = data[data.symptom_class == "Normal"]
plt.scatter(abnormal.lumbar_lordosis_angle, abnormal.degree_spondylolisthesis, color = "red",label = "Abnormal")
plt.scatter(normal.lumbar_lordosis_angle, normal.degree_spondylolisthesis, color = "green",label = "Normal")
plt.legend()
plt.xlabel("Lumbar Lordosis")
plt.ylabel("Degree Spondylolisthesis")
plt.show()
data.symptom_class = [1 if each == "Abnormal" else 0 for each in data.symptom_class]
y = data.symptom_class.values
x_ = data.drop(["symptom_class"],axis=1)
x = (x_ - np.min(x_))/(np.max(x_)-np.min(x_)).values
#Split data into Train and Test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state =42)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3) #set K neighbor as 3
knn.fit(x_train,y_train)
predicted_y = knn.predict(x_test)
print("KNN accuracy according to K=3 is :",knn.score(x_test,y_test))
score_array = []
for each in range(1,25):
    knn_loop = KNeighborsClassifier(n_neighbors = each) #set K neighbor as 3
    knn_loop.fit(x_train,y_train)
    score_array.append(knn_loop.score(x_test,y_test))
    
plt.plot(range(1,25),score_array)
plt.xlabel("Range")
plt.ylabel("Score")
plt.show()
knn_final = KNeighborsClassifier(n_neighbors = 15) #set K neighbor as 15
knn_final.fit(x_train,y_train)
predicted_y = knn_final.predict(x_test)
print("KNN accuracy according to K=15 is :",knn_final.score(x_test,y_test))