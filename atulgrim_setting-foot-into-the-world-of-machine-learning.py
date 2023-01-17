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

#Reading the csv file and saving the data into a dataframe named df.

df = pd.read_csv('/kaggle/input/iris/Iris.csv')
#Checking the first 5(by default) rows of the dataframe

df.head()
df.describe() # When given a mix of categorical and numerical data, then By default it describes only the attributes with numerical values. 
df.describe(include ='all') # include ='all' is a parameter used to include all the numerical as well as categorical values.
#Finding the null values in the column

df.isna().sum()
#Count of each kind of flower species by creating a bar Plot using Pandas.

df['Species'].value_counts().plot(kind='bar')
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#Remove the Id column as logically it does not have much worth in prediting the Species. Right!

df = df[df.columns.drop('Id')]
#Finally plotting the pair plot using the seaborn library

sns.pairplot(df)

plt.show()
#Confirming the same by finding the correlation of all the relevant attributes.

df.corr()
#Better way to represent the correlation matrix

# Compute the correlation matrix

corr = df.corr()



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr)
#Adding Annotation - Displaying numbers inside the each cell of the heat map

sns.heatmap(corr, annot = True)
# Since upper half is mirror of the below half of the diagonal, you can remove either of them. 

# Generate a mask for the upper triangle.

mask = np.triu(np.ones_like(corr, dtype=bool)) 

#np.triu is used for Upper triangle of an array.

#np.ones_like: Return an array of ones with the same shape and type as a given array.

sns.heatmap(corr, annot = True, mask=mask)
# Generate a mask for the lower triangle.

mask = np.tril(np.ones_like(corr, dtype=bool)) #np.tril is used for Lower triangle of an array.

sns.heatmap(corr, annot = True, mask=mask)
#Invert Axis

mask = np.triu(np.ones_like(corr, dtype=bool))

ax = sns.heatmap(corr, annot = True, mask=mask)

ax.invert_yaxis()
import pandas_profiling as pp

pp.ProfileReport(df)
# split into train and test sets

X = df.drop(['Species'],axis=1)

y = df['Species']
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics



#Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)



# fit the model

model = LogisticRegression()

model.fit(X_train, y_train)



# evaluate the model

prediction = model.predict(X_test)



print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,y_test))
from sklearn import svm

model = svm.SVC()

model.fit(X_train,y_train) 

prediction=model.predict(X_test)

print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,y_test))
from sklearn.neighbors import KNeighborsClassifier

model=KNeighborsClassifier(n_neighbors=3)

model.fit(X_train,y_train)

prediction=model.predict(X_test)

print('The accuracy of the KNN is',metrics.accuracy_score(prediction,y_test))
from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier()

model.fit(X_train,y_train) 

prediction=model.predict(X_test) 

print('The accuracy of the Decision Tree is ',metrics.accuracy_score(prediction,y_test))
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=20,criterion='entropy',random_state=0)

model.fit(X_train,y_train) 

prediction=model.predict(X_test) 

print('The accuracy of the Random Forest is ',metrics.accuracy_score(prediction,y_test))
petal=df[['PetalLengthCm','PetalWidthCm','Species']]

sepal=df[['SepalLengthCm','SepalWidthCm','Species']]
train_p,test_p=train_test_split(petal,test_size=0.3,random_state=0)  #petals

train_x_p=train_p[['PetalWidthCm','PetalLengthCm']]

train_y_p=train_p.Species

test_x_p=test_p[['PetalWidthCm','PetalLengthCm']]

test_y_p=test_p.Species
train_s,test_s=train_test_split(sepal,test_size=0.3,random_state=0)  #Sepal

train_x_s=train_s[['SepalWidthCm','SepalLengthCm']]

train_y_s=train_s.Species

test_x_s=test_s[['SepalWidthCm','SepalLengthCm']]

test_y_s=test_s.Species
model=svm.SVC()

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

print('The accuracy of the SVM using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model=svm.SVC()

model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

print('The accuracy of the SVM using Sepal is:',metrics.accuracy_score(prediction,test_y_s))
model = LogisticRegression()

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

print('The accuracy of the Logistic Regression using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

print('The accuracy of the Logistic Regression using Sepals is:',metrics.accuracy_score(prediction,test_y_s))
model=DecisionTreeClassifier()

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

print('The accuracy of the Decision Tree using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

print('The accuracy of the Decision Tree using Sepals is:',metrics.accuracy_score(prediction,test_y_s))
model=KNeighborsClassifier(n_neighbors=3) 

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

print('The accuracy of the KNN using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

print('The accuracy of the KNN using Sepals is:',metrics.accuracy_score(prediction,test_y_s))