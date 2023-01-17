# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#We are going to analyse IRIS flower dataset

#import necessary libraries



import seaborn as sns

import matplotlib.pyplot as plt
#importing iris dataset which is inbuilt into the sklearn package



from sklearn.datasets import load_iris



iris = load_iris()

print(iris)
#note the iris dataset is in the form of a dict, hence convert it to a dataframe

data = pd.DataFrame([iris])



#Now check what the data looks like

data
#As we can set the data is not in the right format for us to analyse, so we do further data preparation

#We need to use data, target, target_names and feature_name colums only



#We use np.c_ a function to concatenate iris['data'] and iris['target'] arrays

#for pandas column argument: concat iris['feature_names'] list



irisdata = pd.DataFrame(data=np.c_[iris['data'],iris['target']], columns = iris['feature_names']+['species'])
#Now lets us view the data

irisdata.head()
#Now let us see what species column has 

irisdata['species'].unique()
#so now we assign those 3 unique values to the species names of the flowers to do classification using ML algorithms

#We can use very simple replace method



feature_names = {0:'setosa', 1:'versicolor', 2:'virginica'}

for each in irisdata.species:

    irisdata.species = irisdata.species.replace(feature_names)

irisdata.head()
irisdata.tail()
#Now dataset is in proper format, we will check for missing values

irisdata.info()
#Anotherway to check for missing values is

irisdata.isnull().values.any()
#we will plot a graph to determine sepal length and sepal width using seaborn



fig = sns.scatterplot(data = irisdata, x=irisdata['sepal length (cm)'], y=irisdata['sepal width (cm)'],  hue = irisdata.species )

fig.set_title('sepal length vs sepal width')

fig
#similarly we can check relationship between petal length and petal width



fig = sns.scatterplot(x=irisdata['petal length (cm)'], y=irisdata['petal width (cm)'], data=irisdata,

               hue = irisdata.species )

fig.set_title("petal length vs petal width")

fig
#Now we will check how length and width are distributed



ax1 = sns.distplot(irisdata['sepal length (cm)'],bins = 10)

plt.title("Sepal length")

plt.show(ax1)



ax2 = sns.distplot(irisdata['sepal width (cm)'])

plt.title("Sepal width") 

plt.show(ax2)



ax3 = sns.distplot(irisdata['petal length (cm)'])

plt.title("petal length") 

plt.show(ax3)



ax4 = sns.distplot(irisdata['petal width (cm)'])

plt.title("petal width") 

plt.show(ax4)
#Now we will also see how length and width vary according to species



sns.violinplot(data = irisdata, x='species', y='sepal length (cm)')

plt.show()

sns.violinplot(data = irisdata, x='species', y='sepal width (cm)')

plt.show()

sns.violinplot(data = irisdata, x='species', y='petal length (cm)')

plt.show()

sns.violinplot(data = irisdata, x='species', y='petal width (cm)')

plt.show()
#import all necessary packages to use various classification algorithms



import pandas

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt



from sklearn import model_selection



from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



#various classification algorithmns

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

#check the correlation in the irisdata 



sns.heatmap(irisdata.corr(), annot=True)
#Now let us split the dat into training set and testing set. 

#then pass training data to .fit() method and testing data to predict()

#then check accuracy by passing predicted outcome and actual output



from sklearn.model_selection import train_test_split



train, test = train_test_split(irisdata, test_size=0.3)

print(train.shape)

print(test.shape)
train_X = train[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']]

train_y = train.species



test_X = test[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']]

test_y = test.species
#Now lets us check the data

train_X.head(2)
test_X.head(2)
train_y.head(2)
#Now we use various algorithms and check the accuracy



model = SVC()

model.fit(train_X, train_y)

prediction = model.predict(test_X)

print('The accuracy of SVM is:', accuracy_score(prediction, test_y))
model = LogisticRegression()

model.fit(train_X, train_y)

prediction = model.predict(test_X)

print('The accuracy of logistic regression is:', accuracy_score(prediction, test_y))
model = DecisionTreeClassifier()

model.fit(train_X, train_y)

print('the accuracy of Decision Tree is:', accuracy_score(model.predict(test_X), test_y))
model = KNeighborsClassifier()

model.fit(train_X, train_y)

print('The accuracy of Knearest Alg is:', accuracy_score(model.predict(test_X), test_y))
model = LinearDiscriminantAnalysis()

model.fit(train_X, train_y)

print('The accuracy of Linear Discriminant Analysis Alg is:', accuracy_score(model.predict(test_X), test_y))
model = GaussianNB()

model.fit(train_X, train_y)

print('The accuracy of Gaussian NB Alg is:', accuracy_score(model.predict(test_X), test_y))
model = RandomForestClassifier()

model.fit(train_X, train_y)

print('The accuracy of Random Forest Classifier Alg is:', accuracy_score(model.predict(test_X), test_y))
petal = irisdata[['petal length (cm)','petal width (cm)', 'species']]

sepal = irisdata[['sepal length (cm)', 'sepal width (cm)','species']]
irisdata.columns
train_p, test_p = train_test_split(petal, test_size = 0.3, random_state = 0)

train_x_p = train_p[['petal length (cm)','petal width (cm)']]

train_y_p = train_p.species

test_x_p = test_p[['petal length (cm)','petal width (cm)']]

test_y_p=test_p.species
train_s,test_s=train_test_split(sepal,test_size=0.3,random_state=0)  #Sepal

train_x_s=train_s[['sepal length (cm)', 'sepal width (cm)']]

train_y_s=train_s.species

test_x_s=test_s[['sepal length (cm)', 'sepal width (cm)']]

test_y_s=test_s.species
model = SVC()

model.fit(train_x_p, train_y_p)

prediction = model.predict(test_x_p)

print('The accuracy of SVM is:', accuracy_score(prediction, test_y_p))





model = SVC()

model.fit(train_x_s, train_y_s)

prediction = model.predict(test_x_s)

print('The accuracy of SVM is:', accuracy_score(prediction, test_y_s))
model = LogisticRegression()

model.fit(train_x_p, train_y_p)

prediction = model.predict(test_x_p)

print('The accuracy of Logistic Regression is:', accuracy_score(prediction, test_y_p))





model = LogisticRegression()

model.fit(train_x_s, train_y_s)

prediction = model.predict(test_x_s)

print('The accuracy of Logistic Regression is:', accuracy_score(prediction, test_y_s))
model = DecisionTreeClassifier()

model.fit(train_x_p, train_y_p)

prediction = model.predict(test_x_p)

print('The accuracy of Decision Tree is:', accuracy_score(prediction, test_y_p))





model = DecisionTreeClassifier()

model.fit(train_x_s, train_y_s)

prediction = model.predict(test_x_s)

print('The accuracy of Decision Tree is:', accuracy_score(prediction, test_y_s))
model = KNeighborsClassifier(n_neighbors =3)

model.fit(train_x_p, train_y_p)

prediction = model.predict(test_x_p)

print('The accuracy of KNN is:', accuracy_score(prediction, test_y_p))



model.fit(train_x_s, train_y_s)

prediction = model.predict(test_x_s)

print('The accuracy of KNN is:', accuracy_score(prediction, test_y_s))
model = LinearDiscriminantAnalysis()

model.fit(train_x_p, train_y_p)

prediction = model.predict(test_x_p)

print('The accuracy of Linear Discriminant Analysis is:', accuracy_score(prediction, test_y_p))



model.fit(train_x_s, train_y_s)

prediction = model.predict(test_x_s)

print('The accuracy of Linear Discriminant Analysis is:', accuracy_score(prediction, test_y_s))
model = GaussianNB()

model.fit(train_x_p, train_y_p)

prediction = model.predict(test_x_p)

print('The accuracy of GaussianNB is:', accuracy_score(prediction, test_y_p))



model.fit(train_x_s, train_y_s)

prediction = model.predict(test_x_s)

print('The accuracy of GaussianNB is:', accuracy_score(prediction, test_y_s))
model = RandomForestClassifier()

model.fit(train_x_p, train_y_p)

prediction = model.predict(test_x_p)

print('The accuracy of Random Forest Classifier is:', accuracy_score(prediction, test_y_p))



model.fit(train_x_s, train_y_s)

prediction = model.predict(test_x_s)

print('The accuracy of Random Forest Classifier is:', accuracy_score(prediction, test_y_s))