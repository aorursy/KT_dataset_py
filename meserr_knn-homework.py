import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
#data process
data = pd.read_csv("../input/column_2C_weka.csv")
data.head()
data['class'] = [1 if item == 'Abnormal' else 0 for item in data['class']]
y = data['class'].values
x = data.drop(['class'],axis=1)
#train and test
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)
# knn model
scorelist = {}
for item in range(2,50):
    knn = KNeighborsClassifier(n_neighbors=item)
    
    knn.fit(x_train, y_train)
    prediction = knn.predict(x_test)
    scorelist[item] = knn.score(x_test,y_test)
    
print(max(zip(scorelist.values(), scorelist.keys())))
plt.plot(scorelist.keys(), scorelist.values())
plt.xlabel("k value")
plt.ylabel("accuracy")
plt.show()    
