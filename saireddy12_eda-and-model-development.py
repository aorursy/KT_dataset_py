import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn

import scipy

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")
project_data = pd.read_csv("../input/mobile-price-classification/train.csv")
x=project_data.shape

print("Number of Data points/Observations in train dataset are:-",x[0])

print("Number of Features in train dataset  are:-",x[1])



#sample of our Data

project_data.head(10)
#checking for the null values

project_data.isna().sum()
#basic info like datatype of the features 

project_data.info()
#to find the corelation between the columns

corr=project_data.corr()

fig = plt.figure(figsize=(15,12))

r = sns.heatmap(corr, cmap='Purples')

r.set_title("Correlation ")
#to know more about the features

project_data.describe()
#pie chart representation

x=project_data['dual_sim'].value_counts()

labels='Supports Dualsim: '+str(x[1]),'Does not support Dualsim:- '+str(x[0])

sizes=[x[1],x[0]]

fig1, ax1 = plt.subplots()

ax1.pie(sizes,labels=labels)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
#pie chart representation

x=project_data['four_g'].value_counts()

labels='Supports 4 g: '+str(x[1]),'Does not support 4 g:- '+str(x[0])

sizes=[x[1],x[0]]

fig1, ax1 = plt.subplots()

ax1.pie(sizes,labels=labels)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
#pie chart representation

x=project_data['three_g'].value_counts()

labels='Supports 3 g: '+str(x[1]),'Does not support 3 g:- '+str(x[0])

sizes=[x[1],x[0]]

fig1, ax1 = plt.subplots()

ax1.pie(sizes,labels=labels)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
#pie chart representation

x=project_data['wifi'].value_counts()

labels='Wifi Enabled: '+str(x[1]),'Does not support Wifi:- '+str(x[0])

sizes=[x[1],x[0]]

fig1, ax1 = plt.subplots()

ax1.pie(sizes,labels=labels)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
#pie chart representation

x=project_data['touch_screen'].value_counts()

labels='touchscreen Enables: '+str(x[1]),'Does not support Touchscreen:- '+str(x[0])

sizes=[x[1],x[0]]

fig1, ax1 = plt.subplots()

ax1.pie(sizes,labels=labels)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
sns.countplot(x='fc', data=project_data)

plt.show()
sns.countplot(x='n_cores', data=project_data)

plt.show()
sns.countplot(x='ram', data=project_data)

plt.show()
sns.boxplot(y='clock_speed',x='price_range',data=project_data)

plt.show()
sns.boxplot(x='touch_screen',y='price_range',data=project_data)

plt.show()
sns.boxplot(x='dual_sim',y='price_range',data=project_data)

plt.show()
sns.boxplot(x='four_g',y='price_range',data=project_data)

plt.show()
sns.boxplot(x='three_g',y='price_range',data=project_data)

plt.show()
sns.boxplot(x='wifi',y='price_range',data=project_data)

plt.show()
sns.boxplot(x='blue',y='price_range',data=project_data)

plt.show()
sns.boxplot(x='n_cores',y='price_range',data=project_data)

plt.show()
sns.boxplot(x='fc',y='price_range',data=project_data)

plt.show()
sns.boxplot(x='ram',y='price_range',data=project_data)
# pairwise scatter plot: Pair-Plot.

sns.set_style("whitegrid");

sns.pairplot(project_data,hue='price_range',vars=['n_cores', 'dual_sim','four_g', 'ram','touch_screen','wifi','talk_time','three_g'])

plt.legend()

plt.show() 
# considering only those features that has impact on price_range from our anlysis

x = project_data[['three_g','battery_power','blue','dual_sim','four_g','px_height','px_width','ram','touch_screen','wifi','fc']]

y = project_data['price_range']

print("shape of x train is" ,x.shape)

print("shape of y train is" ,y.shape)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x)

x = scaler.transform(x)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

print("shape of x train is: ",x_train.shape)

print("shape of y test is-" ,y_test.shape)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

model = LogisticRegression()

model.fit(x_train,y_train)

y_train_pred = model.predict(x_train)

y_test_pred = model.predict(x_test)



print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))

print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))
print("Train Confusion Matrix")

from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_train_pred,y_train), annot=True,annot_kws={"size": 16}, fmt='g')

plt.show()
print("Test Confusion Matrix")

from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_test_pred,y_test), annot=True,annot_kws={"size": 16}, fmt='g')

plt.show()
from sklearn.svm import SVC

model = SVC()

model.fit(x_train,y_train)

y_train_pred = model.predict(x_train)

y_test_pred = model.predict(x_test)



print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))

print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))
print("Train Confusion Matrix")

from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_train_pred,y_train), annot=True,annot_kws={"size": 16}, fmt='g')

plt.show()
print("Test Confusion Matrix")

from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_test_pred,y_test), annot=True,annot_kws={"size": 16}, fmt='g')

plt.show()


from sklearn.tree import DecisionTreeClassifier



model = DecisionTreeClassifier(min_samples_split=10)#we use  min sample split value for preventing model from overfitting 

model.fit(x_train,y_train)

y_train_pred = model.predict(x_train)

y_test_pred = model.predict(x_test)



print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))

print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))
print("Train Confusion Matrix")

from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_train_pred,y_train), annot=True,annot_kws={"size": 16}, fmt='g')

plt.show()
print("Test Confusion Matrix")

from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_test_pred,y_test), annot=True,annot_kws={"size": 16}, fmt='g')

plt.show()
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(min_samples_split=10)

model.fit(x_train,y_train)

y_train_pred = model.predict(x_train)

y_test_pred = model.predict(x_test)



print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))

print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))
print("Train Confusion Matrix")

from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_train_pred,y_train), annot=True,annot_kws={"size": 16}, fmt='g')

plt.show()
print("Test Confusion Matrix")

from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_test_pred,y_test), annot=True,annot_kws={"size": 16}, fmt='g')

plt.show()
project_data.columns
#Models Summary

#http://zetcode.com/python/prettytable/

from prettytable import PrettyTable

    

x = PrettyTable()

x.field_names = ["Model","Test Accuracy"]

x.add_row(["Logistic Regression(LR)",81.83])

x.add_row(["Suppoer Vector Classifier(SVC)",86.6])

x.add_row(["Decision Tree Classsifier",82.6])

x.add_row(["Random Forest",85.3])

print(x)