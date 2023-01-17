#Importing the necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix
path = '/kaggle/input/mushroom-classification/mushrooms.csv'



df = pd.read_csv(path)
df.head()
df.describe()
df.info()
#Replacing e with 1

#Replacing  p with 0

df['class'] = df['class'].map({'e':1, 'p':0})
#Note that all our independent varibles are categorical variables

#Using attribute get_dummies to convert conver our category variables into dummy indicator

df_dummies = pd.get_dummies(df, drop_first = True)
df_dummies.head()
#Declaring our target variable as y

#Declaring our independent variables as x

x = df_dummies.drop('class', axis = 1)

y = df_dummies['class']
#Selecting the model

reg = LogisticRegression()
#Splitting our dataset into train and test datasets 

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 24)
#We train the model with x_train and y_train

reg.fit(x_train, y_train)
#Predicting with our already trained model using x_test

y_hat = reg.predict(x_test)
#Mesuring the accuracy of our model

acc = metrics.accuracy_score(y_hat, y_test)

acc
#The intercept for our regression

reg.intercept_
#Coefficient for all our variables

reg.coef_
cm = confusion_matrix(y_hat, y_test)

cm
# Format for easier understanding

cm_df = pd.DataFrame(cm)

cm_df.columns = ['Predicted 0','Predicted 1']

cm_df = cm_df.rename(index={0: 'Actual 0',1:'Actual 1'})

cm_df
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier # for K nearest neighbours

from sklearn import svm #for Support Vector Machine (SVM) 
dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

y1 = dt.predict(x_test)

acc1 = metrics.accuracy_score(y1, y_test)

acc1
kk = KNeighborsClassifier()

kk.fit(x_train,y_train)

y2 = kk.predict(x_test)

acc2 = metrics.accuracy_score(y2, y_test)

acc2
sv = svm.SVC()

sv.fit(x_train,y_train)

y3 = sv.predict(x_test)

acc3 = metrics.accuracy_score(y3, y_test)

acc3
pd.options.display.max_rows = 999

result = pd.DataFrame(data = x.columns, columns = ['Features'])

result['BIAS'] = np.transpose(reg.coef_)

result['odds'] = np.exp(np.transpose(reg.coef_))

result
#T be able to identify our refernce model

df['cap-shape'].unique()