# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

column_3C_weka = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv")
data.head()
data.info()
data.describe()
data.loc[:,'class']  # data['class'] ayni sonucu verir
data.loc[:,data.columns !=  'class' ].head()
color_list = ['red' if i == 'Abnormal' else 'green' for i in data['class']]

pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],

                           c = color_list,

                           figsize = [15,15],

                           diagonal = 'hist',

                           alpha=0.5,s = 200,

                           marker = '@',

                           edgecolor= "orange")

plt.show()
data['class'].value_counts()
sns.countplot('class', data=data)

plt.show()
data.columns
data[['pelvic_incidence','sacral_slope']]
data1 = data[data['class'] == 'Abnormal']

x = np.array(data1.loc[:,'pelvic_incidence']).reshape(-1,1)

y = np.array(data1.loc[:,'sacral_slope']).reshape(-1,1)

plt.figure(figsize=[6,6])

plt.scatter(x=x,y=y)

plt.xlabel('elvic_incidence')

plt.ylabel('sacral_slope')

plt.show()
# import sklearn libary

from sklearn.linear_model import LinearRegression



# Linear Reggression model

reg = LinearRegression()



# Predict space   

predict_space = np.linspace(min(x),max(x)).reshape(-1,1)    # X values



# Fit Line

reg.fit(x,y)



predicted = reg.predict(predict_space)



# R^2

print('R^2 score: ',reg.score(x, y))



# Plot regression line and scatter

plt.plot(predict_space,predicted, color='black', linewidth=3)

plt.scatter(x=x,y=y)

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.show()
reg.predict([[ 1126.14792141]])
avocado = pd.read_csv("../input/avocado-prices/avocado.csv")
avocado.head()
avocado.info()
avocado.corr(method ='kendall')
avocado.drop("Unnamed: 0", axis = 1, inplace=True)
# Visualize with seaborn library

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(avocado.corr(method ='kendall'), annot=True, linewidths=.5, fmt= '.1f',cmap="YlGnBu", ax=ax, cbar_kws={"orientation": "vertical"})

plt.show()
# Years vs Average Price

bdata = avocado[["year","AveragePrice"]].groupby(["year"],as_index =False).mean().sort_values(by="year", ascending = True)

plt.figure(figsize=(9,3))

plt.bar(bdata["year"].values, bdata["AveragePrice"].values)

plt.xticks(bdata["year"].values)

plt.title("AveragePrice per Years")

plt.show()
# %% linear regression

x = avocado.loc[:,"Total Volume"].values.reshape(-1,1)

y = avocado.loc[:,"AveragePrice"].values.reshape(-1,1)



from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)



#predict

predict_space = np.linspace(min(x),max(x)).reshape(-1,1)

predicted = lr.predict(predict_space)

y_head = lr.predict(x)





plt.scatter(x=x,y=y_head)

plt.plot(predict_space ,predicted ,color="red",label ="linear")

plt.xlabel('Total Volume')

plt.ylabel('AveragePrice')

plt.show()
# Predicts

print(lr.predict([[1000]]))
# x = Total Volume, y = AveragePrice



## Polynomial Linear Regression with 2nd degree

# y =b0 + b1.x + b2.x^2

x = avocado.loc[:,"Total Volume"].values.reshape(-1,1)

y = avocado.loc[:,"AveragePrice"].values.reshape(-1,1)





from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = PolynomialFeatures(degree = 2)

x_polynomial = polynomial_regression.fit_transform(x)



# Line fit

linear_regression2 = LinearRegression()

linear_regression2.fit(x_polynomial,y)



# Visualization

y_head = linear_regression2.predict(x_polynomial)



plt.plot(x,y_head,color= "green",label = "Polynomial")

plt.scatter(x=x,y=y)

plt.xlabel('Total Volume')

plt.ylabel('AveragePrice') 

plt.legend()

plt.show()
multidata = avocado[['Total Volume','4046','4225','4770','AveragePrice']]
multidata.head()
y = multidata["AveragePrice"].values.reshape(-1,1)

x = multidata.iloc[:,[0,1,2,3]]    # x are "Total Volume","4046", "4225", "4770"



multiple_linear_regression = LinearRegression()

multiple_linear_regression.fit(x,y)



y_head = multiple_linear_regression.predict(x)

plt.plot(x,y_head)

plt.legend()

plt.xlabel('Total Volume, 4046, 4225, 4770')

plt.ylabel('AveragePrice')

plt.show()



print("b0: ", multiple_linear_regression.intercept_)

print("b1,b2, b2, b3 : ",multiple_linear_regression.coef_)
multiple_linear_regression.predict([[6.423662, 1.036740, 5.445485, 4.81600]])
x = multidata.iloc[:,[0]].values.reshape(-1,1)

y = multidata.iloc[:,[4]].values.ravel()



from sklearn.tree import DecisionTreeRegressor



# Decision Tree Regression

tree_reg = DecisionTreeRegressor()

tree_reg.fit(x,y)



# Predict

y_ = tree_reg.predict(x)



# Visualization

plt.scatter(x,y,color="red")

plt.plot(x,y_,color="black")

plt.xlabel("Total Avocado Volume")

plt.ylabel("Avarage Price")

plt.show()



tree_reg.predict([[14.3]])
from sklearn.metrics import r2_score

print("r_score: ", r2_score(y,y_))
x = multidata.iloc[:,0].values.reshape(-1,1)

y = multidata.iloc[:,4].values.ravel()



from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 100, random_state = 42) 



# n_estimators = number of tree, how many tree we we are going to use.

rf.fit(x,y)

y_head = rf.predict(x)



#Predict

rf.predict([[1154876.98]])



# Visualize

plt.scatter(x,y,color = "red")

plt.plot(x,y_head,color = "blue")

plt.xlabel("Total Avocado Volume")

plt.ylabel("Avarage Price")

plt.show()



# Random Forest Algoristmasi r-score hesaplama



from sklearn.metrics import r2_score

print("r_score: ", r2_score(y,y_head))
x_l = np.load('/kaggle/input/sign-language-digits-dataset/X.npy')

Y_l = np.load('/kaggle/input/sign-language-digits-dataset/Y.npy')

# 0 - 204 are the numbers equal to 3

# 1236 - 1442 are the numbers equal to 4

plt.imshow(x_l[1237])

plt.show()

three = []

for i in range(205):

    three.append(3)

three = np.array(three)

four = []

for i in range(205):

    four.append(4)

four = np.array(four)
X = np.concatenate((x_l[0:205], x_l[1236:1441] ), axis=0) # from 0 to 204 is three sign and from 205 to 410 is four sign

Y = np.concatenate((three,four), axis=0).reshape(-1,1)

print(" X shape {}, Y shape {}".format(X.shape,Y.shape))
# Then lets create x_train, y_train, x_test, y_test arrays

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
print("number_of_train : {} and number_of_test : {}".format(X_train.shape[0],X_test.shape[0]))

print("X_train.shape : {}".format(X_train.shape))

print("X_test.shape  : {}".format(X_test.shape))

print("Y_train.shape : {}".format(Y_train.shape))

print("Y_test.shape  : {}".format(Y_test.shape))
X_train_flatten = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])

X_test_flatten = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])

print("X train flatten",X_train_flatten.shape)

print("X test flatten",X_test_flatten.shape)
x_train = X_train_flatten

x_test = X_test_flatten

y_train = Y_train

y_test = Y_test

print("x train: ",x_train.shape)

print("x test: ",x_test.shape)

print("y train: ",y_train.shape)

print("y test: ",y_test.shape)
import warnings

warnings.filterwarnings("ignore")

test_accuracy = []

train_accuracy = []

test_accuracy_decent = []

index_decent =[]

index = []

from sklearn import linear_model

for i in range(150):

    logreg = linear_model.LogisticRegression(random_state = 42,max_iter= i )

    test_accuracy.append(logreg.fit(x_train, y_train).score(x_test, y_test))

    train_accuracy.append(logreg.fit(x_train, y_train).score(x_train, y_train))

    index.append(i)

    if i % 10 == 0:

        test_accuracy_decent.append(logreg.fit(x_train, y_train).score(x_test, y_test))

        index_decent.append(i)







plt.plot(index_decent,test_accuracy_decent)

plt.xticks(index_decent)

plt.xlabel("Number of Iterarion")

plt.ylabel("Test Accuracy")

plt.show()
print("Test Accurcy is : {}".format(max(test_accuracy_decent)))
data = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
data.head()
# We have 210 Abnormal and 100 Normal samples in the dataset.

data["class"].value_counts()
A = data[data["class"] == "Abnormal"]

N = data[data["class"] == "Normal"]

# scatter plot

plt.scatter(A.pelvic_incidence,A["degree_spondylolisthesis"],color="red",label="Abnormal",alpha= 0.3)

plt.scatter(N.pelvic_incidence,N["degree_spondylolisthesis"],color="green",label="Normal",alpha= 0.3)

plt.xlabel("pelvic_incidence")

plt.ylabel("degree_spondylolisthesis")

plt.legend()

plt.show()
data["class"] = [1 if each == "Abnormal" else 0 for each in data["class"]]
data.head()
y = data["class"].values

x_data = data.drop(["class"],axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

print(" {} knn score: {} ".format(3,knn.score(x_test,y_test)))
score_list = []

for each in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,15),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
data.head()
# SVM Classification

from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(x_train,y_train)

print("print accuracy of svm algo: ",svm.score(x_test,y_test))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print("print accuracy of naive bayes algo: ",nb.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)



print("score: ", dt.score(x_test,y_test))
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators = 100,random_state = 1)

rf.fit(x_train,y_train)

print("random forest algo result: ",rf.score(x_test,y_test))
score_rf = []

for each in range(1,100):

    rf2 = RandomForestClassifier(n_estimators = each,random_state = 1)

    rf2.fit(x_train,y_train)

    score_rf.append(rf2.score(x_test,y_test))

    

plt.plot(range(1,100),score_rf)

plt.xlabel("estimators")

plt.ylabel("accuracy")

plt.show()
print('Random Forest max Accucancy is {}'.format(max(score_rf)))
for i in range(len(score_rf)):

    if pd.DataFrame(score_rf).values[i] == max(pd.DataFrame(score_rf).values):

        print('Max Accuracy Estimater', i)