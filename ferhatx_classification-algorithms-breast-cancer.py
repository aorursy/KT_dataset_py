# Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
data = pd.read_csv('../input/breastCancer.csv')
data.head()
# Clear the noisy attributes
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data.tail()
M = data[data.diagnosis=='M']
B = data[data.diagnosis=='B']
plt.scatter(M.radius_mean,M.texture_mean,color='red',label='Malignant',alpha=0.3)
plt.scatter(B.radius_mean,B.texture_mean,color='green',label='Benign',alpha=0.3)
plt.xlabel('Malignant')
plt.ylabel('Benign')
plt.legend()
plt.show()
# Change M and B values to 0 and 1
# Prepare x and y values for KNN algorithm
data.diagnosis= [1 if each=="M" else 0 for each in data.diagnosis]
y=data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)
# Normalization
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
# Train-Test Split for Learning
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3) #k=3
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("{} nn score: {}".format(3,knn.score(x_test,y_test)))
# Hyperparameter Tuning
score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()
from sklearn.svm import SVC
svm = SVC(random_state=1) # Return the same value every time
svm.fit(x_train,y_train)

# test
print("primy accuracy of SVM algorithm : ",svm.score(x_test,y_test))
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

#test
print("Accuracy of Naive-Bayes Algorithm",nb.score(x_test,y_test))
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=42)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
# Accuracy
print("Accuracy of Decision Tree Algorithm",dt.score(x_test,y_test))
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,random_state=1) # Number of tree = 100
rf.fit(x_train,y_train)
print("Accuracy of Random Forest Algorithm",rf.score(x_test,y_test))
accuracy_list=[]
for i in range(1,11,1):
    rf = RandomForestClassifier(n_estimators=i,random_state=1) # Number of tree = 100
    rf.fit(x_train,y_train)
    accuracy_list.append(rf.score(x_test,y_test))
    #print("Accuracy of Random Forest Algorithm for {} trees: {}".format(i,rf.score(x_test,y_test)))
plt.plot(range(1,11),accuracy_list)
plt.xlabel("Number of estimators")
plt.ylabel("Accuracy")
plt.show()