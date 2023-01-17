import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

df=pd.read_csv("../input/classified-data/Classified Data",index_col=0)

df.head()

#The columns in the data mean nothing so it is better to use K mearest Neighbors algorithm
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler() #here we create instance of Standart Scaler
scaler.fit(df.drop("TARGET CLASS",axis=1)) 

# here we standartize our features in our dataset apart from the target column
#After fitting with our data we will transform our data according to Standart Scaler

scaled_features=scaler.transform(df.drop("TARGET CLASS",axis=1))

scaled_features 

# here we get standartized features of our dataset as numpy arrays
df_features=pd.DataFrame(scaled_features,columns=df.columns[:-1])

df_features.head()

# Now all of the features has been standartized and is ready to be put into machine learning algorithm
#We will split our data before training the algorith:

X=df_features

y=df["TARGET CLASS"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train) # the algorithm fits with our data
predictions=knn.predict(X_test)

predictions
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,predictions))



#TP=151 : true positive

#FN=8   : false negative

#FP=15  :false positive 

#TN=126 : true negative

#The errors are not too high and absorable

print(classification_report(y_test,predictions))



#The precision and accuracy precentages are over %90, it is very good with k=1
#Although k=1 is very good for our predcitions, we will check whether there is better k value or not

error_rate=list()

#here we iterate meny different k values and plot their error rates 

#and discover which one is better than others and has the lowest error rate

for i in range(1,40):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    prediction_i=knn.predict(X_test)

    error_rate.append(np.mean(prediction_i != y_test))
# Now we will plot the prediction error rates of different k values

plt.figure(figsize=(15,10))

plt.plot(range(1,40),error_rate, color="blue", linestyle="--",marker="o",markerfacecolor="red",markersize=10)

plt.title("Error Rate vs K Value")

plt.xlabel="K Value"

plt.ylabel("Error Rate")

knn=KNeighborsClassifier(n_neighbors=34)

knn.fit(X_train, y_train)

predictions=knn.predict(X_test)

print(classification_report(y_test,predictions))

print("\n")

print(confusion_matrix(y_test,predictions))