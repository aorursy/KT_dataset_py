#let us read the data
import pandas as pd

data = pd.read_csv('../input/breastCancer.csv')
data.shape
data.head(10)
y = data.iloc[:,1]
#the values are categorical so we need to transform them
y[:10]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
y[:10]
X = data.iloc[:,2:29]
import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=7)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
X_test
X_train.head(10)
#Import Library
#Import other necessary libraries like pandas, numpy...
from sklearn import linear_model
# Create linear regression object
linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(X_train, y_train)
linear.score(X_train, y_train)
#Equation coefficient and Intercept
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
#Predict Output
predicted = linear.predict(X_test)
z = [int(i)for i in predicted] #these are the values predicted we take either 0 or 1

Z = pd.DataFrame(z)
Z
X_test.info()
#accuracy
from sklearn.metrics import accuracy_score
y = accuracy_score(y_test,Z)
y
#if I use the full dataset of X which is not sliced I can be able to get the values of all the inputs
w = linear.predict(X)
#turn the values into dataframes
w1 = [int(i) for i in w]
w2 = pd.DataFrame(w1)
w2.head(10)
#i want to add the values to their IDs and the original diagnosis
labels = ['Orig','Predicted','ID']
l1 = data.iloc[:,1]
l2 = w2.iloc[:,0]
l3 = data.iloc[:,0]
columns = [l1,l2,l3]
# for i, j in zip(l1,l2):
#     print(i,j)
df = pd.DataFrame(columns,labels)
df.T.head(10)


X2 = X.copy() #this is fine for our tex
X2
y2 = data.iloc[:,1]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y2 = le.fit_transform(y2)
#split the data
from sklearn.model_selection import train_test_split

X2_train,X2_test,y2_train,y2_test, = train_test_split(X2,y2,test_size = 0.25, random_state = 7)
#Import Library
from sklearn.linear_model import LogisticRegression
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create logistic regression object
model = LogisticRegression()
# Train the model using the training sets and check score
model.fit(X2, y2)
model.score(X2, y2)
#Equation coefficient and Intercept
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)
#Predict Output
predicted2 = model.predict(X2_test)
predicted2
predicted2
accuracy_score(y2_test,predicted2)#calculate the accuracy score, and its better than logistics regression
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y2_test,predicted2))
from sklearn.metrics import classification_report
print(classification_report(y2_test,predicted2))


X3 = X2.copy()
y3 = y2.copy()
from sklearn.model_selection import train_test_split

X3_train, X3_test, y3_train, y3_test = train_test_split(X3,y3, test_size = 0.25,random_state = 10)
X3_train.shape, X3_test.shape, y3_train.shape, y3_test.shape
#Import Library
#Import other necessary libraries like pandas, numpy...
from sklearn import tree
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create tree object 
model = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
# model = tree.DecisionTreeRegressor() for regression
# Train the model using the training sets and check score
model.fit(X3, y3)
model.score(X3, y3)
#Predict Output
predicted3= model.predict(X3_test)
predicted3
accuracy_score(y3_test, predicted3)

X4 = X2.copy()
X4.head(10)
y4 = y2.copy()
y4[:10]
from sklearn.model_selection import train_test_split
X4_test, X4_train, y4_test, y4_train = train_test_split(X4,y4,test_size = 0.30,random_state = 10)
#Import Library
from sklearn.svm import SVC
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 
model = SVC(kernel = 'linear') # there is various option associated with it, this is simple for classification. You can refer link, for mo# re detail.
# Train the model using the training sets and check score
model.fit(X4, y4)
model.score(X4, y4)
#Predict Output
predicted4= model.predict(X4_test)
predicted4[:10]
accuracy_score(y4_test, predicted4)
from sklearn.metrics import classification_report
print(classification_report(y4_test, predicted4 ))

X5 = X2.copy()
y5 = y2.copy()
from sklearn.model_selection import train_test_split

X5_train, X5_test, y5_train, y5_test = train_test_split(X5,y5,test_size = 0.30 ,random_state = 10)
X5_train.shape, X5_test.shape, y5_train.shape, y5_test.shape
#Import Library
from sklearn.naive_bayes import GaussianNB
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object model = GaussianNB() # there is other distribution for multinomial classes like Bernoulli Naive Bayes, Refer link
# Train the model using the training sets and check score
model.fit(X5, y5)
#Predict Output
predicted5 = model.predict(X5_test)
predicted5
from sklearn.metrics import accuracy_score
accuracy_score(y5_test,predicted5)
from sklearn.metrics import average_precision_score
average_precision_score(y5_test,predicted5)
from sklearn.metrics import classification_report
u =classification_report(y5_test,predicted5)
print(u)

X6 = X2.copy()
y6 = y2.copy()
X6_train, X6_test, y6_train, y6_test = train_test_split(X6,y6, test_size = 0.30, random_state = 5)
X6_train.shape, X6_test.shape, y6_train.shape, y6_test.shape
#Import Library
from sklearn.neighbors import KNeighborsClassifier
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create KNeighbors classifier object model 
KNeighborsClassifier(n_neighbors=6) # default value for n_neighbors is 5
# Train the model using the training sets and check score
model.fit(X6, y6)
#Predict Output
predicted6= model.predict(X6_test)
accuracy_score(y6_test, predicted6)

X7 = X2.copy()
y7 = y2.copy()
from sklearn.model_selection import train_test_split
X7_train, X7_test, y7_train, y7_test = train_test_split(X7,y7, test_size = 0.30, random_state = 0)
# #checking the optimak number of clusters
# from sklearn.decomposition import PCA 
# pca = PCA(n_components=8)
# fit = pca.fit(X7)
# fit
# import matplotlib.pyplot as plt
# val = pca.explained_variance_ratio_
# plt.plot(val)
# plt.show()
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=5,random_state= 0)
fit = k_means.fit(X7)
k_means.predict(X7_test)
#Another way to do it is as below:-
from sklearn.neighbors import KNeighborsClassifier
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create KNeighbors classifier object model 
KNeighborsClassifier(n_neighbors=6) # default value for n_neighbors is 5
# Train the model using the training sets and check score
model.fit(X7, y7)
#Predict Output
predicted7= model.predict(X7_test)
#testing accuracy of the module
from sklearn.metrics import adjusted_rand_score
adjusted_rand_score(y7_test,predicted7 )
x8 = X2.copy()
y8 = y2.copy()
from sklearn.model_selection import train_test_split
x8_train, x8_test, y8_train, y8_test = train_test_split(x8,y8, test_size = 0.30, random_state = 5)
x8_train.shape, x8_test.shape, y8_train.shape, y8_test.shape
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x8,y8)
predicted8 = model.predict(x8_test)
accuracy_score(y8_test,predicted8)
print(classification_report(y8_test,predicted8))
