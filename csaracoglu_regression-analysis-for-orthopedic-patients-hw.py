# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

from sklearn.linear_model import LinearRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#read data

data = pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

data.info()

data.describe()
data.head()
sns.countplot(x="class", data=data)

data.loc[:,'class'].value_counts()
#Correlation map

data.corr() 

f, ax = plt.subplots(figsize = (8,8))

sns.heatmap(data.corr(), annot = True, linewidths=.5, fmt = ".2f", ax=ax)

plt.show()
#Dividing my data as Normal and Abnormal class

data_a = data[data["class"] == "Abnormal"]

data_n = data[data["class"] == "Normal"]



#For Abnormal data

x_a = np.array(data_a.loc[:,'pelvic_incidence']).reshape(-1,1)

y_a = np.array(data_a.loc[:,'sacral_slope']).reshape(-1,1)

#For Normal data

x_n = np.array(data_n.loc[:,'pelvic_incidence']).reshape(-1,1)

y_n = np.array(data_n.loc[:,'sacral_slope']).reshape(-1,1)



# Scatter plot

plt.scatter(x = x_a, y = y_a, color = "red", label = "Abnormal", alpha = 0.5)

plt.scatter(x = x_n, y = y_n, color = "green", label = "Normal", alpha = 0.5)

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.legend()

plt.title("pelvic_incidence and sacral_slope for Normal and Abnormal class")

plt.show()
#Linear Regression and drawing line and calculate accuracy

lr = LinearRegression()

lr.fit(x_a, y_a)

predicted_a = np.linspace(min(x_a), max(x_a)).reshape(-1,1)

y_a_head = lr.predict(predicted_a)

print('Accuracy for Abnormal: ',lr.score(x_a, y_a))



lr2 = LinearRegression()

lr2.fit(x_n, y_n)

predicted_n = np.linspace(min(x_n), max(x_n)).reshape(-1,1)

y_n_head = lr2.predict(predicted_n)

print('Accuracy for Normal: ',lr2.score(x_n, y_n))



# Plot regression line and scatter

plt.figure(figsize=[10,8])

plt.plot(predicted_a, y_a_head, color='black', linewidth=3, label = "Abnormal", linestyle='dashed')

plt.plot(predicted_n, y_n_head, color='black', linewidth=3, label = "Normal")

plt.scatter(x = x_a, y = y_a, color = "red", label = "Abnormal", alpha = 0.5)

plt.scatter(x = x_n, y = y_n, color = "green", label = "Normal", alpha = 0.5)

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.legend()

plt.title("pelvic_incidence and sacral_slope for Normal and Abnormal class")

plt.show()
data.head()
data.columns = data.columns.str.replace('class','bone_type')#I chance class because it has some refer problem



#Convert Abnormal to 1 Normal 0 for binary representation

data.bone_type = [1 if each == "Abnormal" else 0 for each in data.bone_type]



y = data.bone_type.values #take all values into y 

x_data = data.drop(['bone_type'], axis = 1)



#Normalization for all values into x between 0 and 1 for calculation

#  (x- min(x)) / (max(x) - min(x))

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values



#Split data for train and test and test size is assumed as 20%

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)



print("x_train: ", x_train.shape,"y_train: ", y_train.shape," \nx_test: ", x_test.shape, " y_test: ", y_test.shape)
from sklearn.linear_model import LogisticRegression



lor = LogisticRegression()

lor.fit(x_train,y_train)



y_predict = lor.predict(x_test)



def find_accuracy(y_test, y_predict):

    count = 0

    if(len(y_test) == len(y_predict)):

        for i in range(len(y_test)):

            if(y_test[i] == y_predict[i]): 

                count += 1

        acc = count / len(y_test) * 100

        print("Accuracy -->", acc, "%")

    else:

        print("Your test and predicted data set are not equal accuracy will not be calculated")



find_accuracy(y_test, y_predict)

#lor.score(x_test, y_test)



x_1 = np.array(x_data.pelvic_incidence).reshape(-1,1)

y_1 = np.array(data.bone_type.values).reshape(-1,1)



#Linear Regression

x_n = (x_1 - min(x_1)) / (max(x_1) - min(x_1))

lr = LinearRegression()

lr.fit(x_n,y_1)

lp = np.linspace(min(x_n), max(x_n)).reshape(-1,1)

y_head = lr.predict(lp)

print('Accuracy for Linear Regression: ',lr.score(x_n, y_1))



#Logistic Regression

lgr = LogisticRegression()

lgr.fit(x_n, y_1)

lp = np.linspace(min(x_n), max(x_n)).reshape(-1,1)

y_head2 = lgr.predict(lp)

print('Accuracy Logistic Regression: ',lgr.score(x_n, y_1))



# Plot regression line and scatter

plt.plot(lp, y_head, color='black', linewidth=3, label = "Linear")

plt.plot(lp, y_head2, color='red', linewidth=3, label = "Logistic")

plt.scatter(x = x_n, y = y_1)

plt.xlabel("Pelvic Incidence")

plt.ylabel("Type(Normal || Abnormal)")

plt.legend()

plt.title("Linear and Logistic Regression")

plt.show()



#For plotting I need to change column[class] name because class is defined word.

#Then I will represent my class attribitues as binary representation 

a_data = data[data["bone_type"] == 1]

n_data = data[data["bone_type"] == 0]



plt.scatter(a_data.degree_spondylolisthesis, a_data.lumbar_lordosis_angle, color = "red", label = "Abnormal", alpha=0.5)

plt.scatter(n_data.degree_spondylolisthesis, n_data.lumbar_lordosis_angle, color = "green", label = "Normal", alpha = 0.5)

plt.xlabel("Pelvic Incidence")

plt.ylabel("Pelvic Radius")

plt.legend()

plt.title("Pelvic Incidence and Radius for Normal and Abnormal Bone Type")

plt.show()



x_all = data.drop(["bone_type"], axis = 1) # all values except bone type

y = data.bone_type.values #label 0 or 1



#Normalization

x_norm = (x_all - np.min(x_all))/(np.max(x_all)-np.min(x_all))
#Train-Test Splitting

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size = 0.3, random_state = 42)
#KNN Algorithm application

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5) # give number of neighbor as 5

knn.fit(x_train, y_train)

prediction = knn.predict(x_test)

print("KNN Accuracy: ", knn.score(x_test, y_test))



#Find best neighbor number for this situation and then plot it

accuracy_list = []

for each in range (1,30):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train, y_train)

    accuracy_list.append(knn2.score(x_test, y_test))



#Plot with line diagram

plt.plot(range(1,30),accuracy_list, label = "Accuracy", linewidth = 3)

plt.grid()

plt.xlabel("K Neighbors numbers")

plt.ylabel("Accuracy")

plt.title("Accuracy graph according to Neighbors numbers")

plt.show()


