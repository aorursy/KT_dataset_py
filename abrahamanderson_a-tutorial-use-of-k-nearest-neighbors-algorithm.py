# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np
df=pd.read_csv("../input/KNN_Data")

df.head()
sns.pairplot(df,hue="TARGET CLASS")

#here we will vizualize all the mutual relations between column and the hue value will be the target column
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler() # here we create an object if Standart Scaler
scaler.fit(df.drop("TARGET CLASS",axis=1))

#We will only fit the scaler for the features not for the target column, so we just drop it
scaled_features=scaler.transform(df.drop("TARGET CLASS",axis=1))

# here we standartize our features in our dataset apart from the target column

# the .transform() method to transform the features to a scaled version

scaled_features
df_features=pd.DataFrame(scaled_features,columns=df.columns[:-1])

df_features.head() #the features are now a pandas dataframe
#We will split our data before training the algorithm:

X=df_features

y=df["TARGET CLASS"]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=101)
#Here we call the algorithm we will use and make the algorithm fit with our training dataset

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1) 

# we select 1 as our k value, we will start with one and change it after evaluation ig it fails to predict the target variable

knn.fit(X_train,y_train)  # the algorithm fits with our data
predictions=knn.predict(X_test)

predictions
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
error_rate=list() #here we create a list in order to keep record of every k value we will use in the for loop

for i in range(1,100):

    knn=KNeighborsClassifier(n_neighbors=i) 

    knn.fit(X_train,y_train)

    prediction_i=knn.predict(X_test)

    error_rate.append(np.mean(prediction_i != y_test)) 

# here we add the mean of predictions of each k value which does not match with the target variable 
plt.figure(figsize=(20,15))

plt.plot(range(1,100),error_rate,color="blue",linestyle="--",marker="o",markerfacecolor="red",markersize=10)

plt.title("Error Rate and K Value")

plt.xlabel="K Value"

plt.ylabel="Error Rate"
#Here we select k=39 which provides one of lowest error rate as our key value and retrain the algorithm

knn=KNeighborsClassifier(n_neighbors=39)

knn.fit(X_train,y_train)

predictions=knn.predict(X_test)

predictions
print(confusion_matrix(y_test,predictions))

print("\n")

print(classification_report(y_test,predictions))