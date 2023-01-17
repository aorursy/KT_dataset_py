

import pandas as pd

data=pd.read_csv("../input/Iris.csv")

from sklearn.model_selection import train_test_split

data=data.drop(['Id'],axis=1)

X=data.drop(['Species'],axis=1)

y=data['Species']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=4)

#import the KNeighborsClassifier class from sklearn

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

k_range = range(1,50) # checking accuracy for a range of k so that we can choose best model 

scores = {}

scores_list = []

for k in k_range:

        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(X_train,y_train)

        y_pred=knn.predict(X_test)

        scores[k] = metrics.accuracy_score(y_test,y_pred)

        scores_list.append(metrics.accuracy_score(y_test,y_pred))

scores

import matplotlib.pyplot as plt

#plot the relationship between K and the accuracy

plt.plot(k_range,scores_list)

plt.xlabel('Values of K')

plt.ylabel('Accuracy')

# we can choose value of k from 3 to 27 where accuracy is 97.77 % 

# i am choosing k = 5 for the model

knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(X,y)

# predict for new data

new_data=[[3,4,5,6],[7,8,9,10],[1,3.4,5.6,7.8],[3,4,5,2],[5,4,2,2]]

new_predict=knn.predict(new_data)

new_predict