# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #data visualization

import seaborn as sns #data visualization

import numpy as np



import warnings            

warnings.filterwarnings("ignore") 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Load dataset

data=pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')

#data includes how many rows and columns

data.shape

print("Our data has {} rows and {} columns".format(data.shape[0],data.shape[1]))

#Features name in data

data.columns
#diplay first 5 rows

data.head()
data.drop(['RowNumber', 'CustomerId', 'Surname','Geography'], axis=1, inplace=True)



#I replaced Gender feature from Male/Female to 1/0.

data.Gender = [1 if each == 'Male' else 0 for each in data.Gender] 
data.describe().T
#checking for missing values

print('Are there missing values? {}'.format(data.isnull().any().any()))

#missing value control in features

data.isnull().sum()
plt.figure(figsize=[5,5])

sns.set(style='darkgrid')

ax = sns.countplot(x='Exited', data=data, palette='Set2')

data.loc[:,'Exited'].value_counts()
y = data.Exited.values

x_data = data.drop(['Exited'], axis=1)
#we should normalize our features, features should dominate each other.

x = (x_data - np.min(x_data)) / (np.max(x_data)-np.min(x_data))

x.head()
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)



print('x_train shape: ', x_train.shape)

print('y_train shape: ', y_train.shape)

print('x_test shape: ', x_test.shape)

print('y_test shape: ', y_test.shape)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_train,y_train)



y_pred=lr.predict(x_test)



from sklearn.metrics import classification_report,confusion_matrix

lr_cm = confusion_matrix(y_test, y_pred)

print("confusion matrix:\n",lr_cm)



print('test accuracy: {}'.format(lr.score(x_test,y_test)))

print('Classification report: \n',classification_report(y_test,y_pred))
#find k value

from sklearn.neighbors import KNeighborsClassifier



score_list=[]

for each in range(1,15):

    knn2=KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,15),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
#knn algorithm



knn=KNeighborsClassifier(n_neighbors=13) #n_neighbors=k

knn.fit(x_train,y_train)

prediction=knn.predict(x_test)



from sklearn.metrics import confusion_matrix

knn_cm = confusion_matrix(y_test, prediction)

print("confusion matrix:\n",knn_cm)



print('test accuracy: {}'.format(knn.score(x_test,y_test)))

print('Classification report: \n',classification_report(y_test,y_pred))
#svm algorithm

from sklearn.svm import SVC

svm = SVC(random_state=0)

svm.fit(x_train,y_train)

prediction=svm.predict(x_test)



from sklearn.metrics import confusion_matrix

svm_cm = confusion_matrix(y_test, prediction)

print("confusion matrix:\n",svm_cm)



print('test accuracy: {}'.format(svm.score(x_test,y_test)))

print('Classification report: \n',classification_report(y_test,y_pred))
#naive bayes algorithm

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

prediction=nb.predict(x_test)



from sklearn.metrics import confusion_matrix

nb_cm = confusion_matrix(y_test, prediction)

print("confusion matrix:\n",nb_cm)



print('test accuracy: {}'.format(nb.score(x_test,y_test)))

print('Classification report: \n',classification_report(y_test,y_pred))
#desicion tree algorithm

from sklearn.tree import DecisionTreeClassifier

cart = DecisionTreeClassifier()

cart.fit(x_train,y_train)

prediction=cart.predict(x_test)



from sklearn.metrics import confusion_matrix

cart_cm = confusion_matrix(y_test, prediction)

print("confusion matrix:\n",cart_cm)



print('test accuracy: {}'.format(cart.score(x_test,y_test)))

print('Classification report: \n',classification_report(y_test,y_pred))
#desicion tree algorithm

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=3)

rf.fit(x_train,y_train)

prediction=rf.predict(x_test)



from sklearn.metrics import confusion_matrix

rf_cm = confusion_matrix(y_test, prediction)

print("confusion matrix:\n",rf_cm)



print('test accuracy: {}'.format(rf.score(x_test,y_test)))

print('Classification report: \n',classification_report(y_test,y_pred))
fig = plt.figure(figsize=(15,15))



ax1 = fig.add_subplot(3, 3, 1) # row, column, position

ax1.set_title('Logistic Regression Classification')



ax2 = fig.add_subplot(3, 3, 2) # row, column, position

ax2.set_title('KNN Classification')



ax3 = fig.add_subplot(3, 3, 3)

ax3.set_title('SVM Classification')



ax4 = fig.add_subplot(3, 3, 4)

ax4.set_title('Naive Bayes Classification')



ax5 = fig.add_subplot(3, 3, 5)

ax5.set_title('Decision Tree Classification')



ax6 = fig.add_subplot(3, 3, 6)

ax6.set_title('Random Forest Classification')





sns.heatmap(data=lr_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax1, cmap='BuPu')

sns.heatmap(data=knn_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax2, cmap='BuPu')   

sns.heatmap(data=svm_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax3, cmap='BuPu')

sns.heatmap(data=nb_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax4, cmap='BuPu')

sns.heatmap(data=cart_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax5, cmap='BuPu')

sns.heatmap(data=rf_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax6, cmap='BuPu')

plt.show()