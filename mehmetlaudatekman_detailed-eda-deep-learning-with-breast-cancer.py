# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



"""

Data Manipulating

"""

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



"""

Visualization

"""

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style("darkgrid")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
data.head()
data.info()
fig,ax = plt.subplots(figsize=(8,6))

sns.countplot(data.diagnosis)

plt.show()
fig,ax = plt.subplots(figsize=(8,6))

sns.distplot(data["radius_mean"],color="#FC2D2D")

plt.show()
fig,ax = plt.subplots(figsize=(8,6))

sns.distplot(data["texture_mean"],color="#F739F7")

plt.show()
fig,ax = plt.subplots(figsize=(8,6))

sns.distplot(data["smoothness_mean"],color="#F5BA5D")

plt.show()
def outlier_index_detector(df,features):

    indexes = []

    result = []

    for ftr in features:

        

        Q1 = df.describe()[ftr]["25%"] # Lower quartile

        Q3 = df.describe()[ftr]["75%"] # Upper quartile

        IQR = Q3 - Q1 # IQR

        STEP = IQR*1.5 # Outlier Step

        

        ind = data[(data[ftr]<Q1-STEP) | (data[ftr]>Q3+STEP)].index.values

        for i in ind:

            indexes.append(i)

    

    for index in indexes:

        

        indexes.remove(index) 

        if index in indexes: # More than 2

            indexes.remove(index)

            

            if index in indexes: # More than 3

                indexes.remove(index)

                

                if index in indexes: # More than 4

                    indexes.remove(index)

                    

                    if index in indexes: # Append Final Result

                        result.append(index)

            

    

    return result

    
feature_names = (list(data))

feature_names.remove("id")

feature_names.remove("diagnosis")

feature_names.remove("Unnamed: 32")

print(feature_names)
outliers = outlier_index_detector(data,feature_names)

print(outliers)
outliers = list(np.unique(outliers))

print("There are {} outlier rows \n".format(len(outliers)))

print(outliers)
print("Len of the dataset before dropping outliers",len(data))

data.drop(outliers,inplace=True)

print("Len of the dataset after dropping outliers",len(data))
fig,ax = plt.subplots(figsize=(20,20))

sns.heatmap(data.corr(),annot=True,linewidths=1.5,fmt="0.1f")

plt.show()
fig,ax = plt.subplots(figsize=(10,8))

sns.scatterplot(x="radius_mean",y="texture_mean",data=data,color="#670F91")

plt.show()
fig,ax = plt.subplots(figsize=(10,8))

sns.scatterplot(x="radius_mean",y="smoothness_mean",data=data,color="#BD6F4B")

plt.show()
data.drop(["Unnamed: 32","id"],axis=1,inplace=True)
print("First 5 entries",data.diagnosis[:5])

data.diagnosis = [0 if each == "M" else 1 for each in data.diagnosis]

print(data.diagnosis[:5])
data.tail()
data = (data-np.min(data)) / (np.max(data)-np.min(data))

data.head()
from sklearn.model_selection import train_test_split

x = data.drop("diagnosis",axis=1) 

y = data.diagnosis



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
print("Len of the x_train",len(x_train))

print("Len of the x_test ",len(x_test))

print("Len of the y_train",len(y_train))

print("Len of the y_test ",len(y_test))
from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential

from keras.layers import Dense
def build_classifier():

    classifier = Sequential()

    classifier.add(Dense(units=12,kernel_initializer="uniform",activation="tanh",input_dim=30))

    classifier.add(Dense(units=6,kernel_initializer="uniform",activation="tanh"))

    classifier.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid")) # Output Layer

    classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

    return classifier

                   
classifier = KerasClassifier(build_fn=build_classifier,epochs=100)

from sklearn.model_selection import cross_val_score



accuracies = cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=3)



print("Mean of CV scores",accuracies.mean())

print("Variance of CV scores",accuracies.std())

classifier.fit(x_train,y_train)
print("Our train score is",classifier.score(x_train,y_train))

print("Our test score is ",classifier.score(x_test,y_test))
