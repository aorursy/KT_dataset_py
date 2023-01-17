# %load Knn.py
# Importing Libraries 
import pandas as pd
import numpy as np


iris = pd.read_csv("../input/iris/Iris.csv")


iris.head()
# Training and Test data using 
from sklearn.model_selection import train_test_split
train,test = train_test_split(iris,test_size = 0.2) # 0.2 => 20 percent of entire data 

# KNN using sklearn 
# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC

# for 3 nearest neighbours 
neigh = KNC(n_neighbors= 3)

# Fitting with training data 
neigh.fit(train.iloc[:,0:5],train.iloc[:,5])

# train accuracy 
train_acc = np.mean(neigh.predict(train.iloc[:,0:5])==train.iloc[:,5])

# test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,0:5])==test.iloc[:,5])



# for 5 nearest neighbours
neigh = KNC(n_neighbors=5)

# fitting with training data
neigh.fit(train.iloc[:,0:5],train.iloc[:,5])

# train accuracy 
train_acc = np.mean(neigh.predict(train.iloc[:,0:5])==train.iloc[:,5])

# test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,0:5])==test.iloc[:,5])

# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values 
 
for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:5],train.iloc[:,5])
    train_acc = np.mean(neigh.predict(train.iloc[:,0:5])==train.iloc[:,5])
    test_acc = np.mean(neigh.predict(test.iloc[:,0:5])==test.iloc[:,5])
    acc.append([train_acc,test_acc])





import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")


plt.legend(["train","test"])
