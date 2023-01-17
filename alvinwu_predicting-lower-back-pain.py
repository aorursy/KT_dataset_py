# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from IPython.display import Image,display

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Column names ~ 12 features, 1 binary class
cols = ['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle','sacral_slope','pelvic_radius',\
        'degree_spondylolisthesis','pelvic_slope','direct_tilt','thoracic_slope','cervical_tilt',\
        'sacrum_angle','scoliosis_slope','normality']

#Read in data using column names, remove last column
data = pd.read_csv('../input/Dataset_spine.csv',header=0,names=cols,usecols=range(0,13))
data.head()
#Initial data stats
print("Number of examples: ",data.shape[0])
print("Number of features: ",data.shape[1]-1)
data.describe()
#Check for missing values 
data.isnull().sum()
#Make a copy of the original data for understanding features
data_f = data.copy()

#Verify PI = PT + SS
data_f['pelvic_tilt + sacral_slope'] = data_f['pelvic_tilt']+data_f['sacral_slope']
cols = ['pelvic_incidence','pelvic_tilt + sacral_slope']
data_f[cols].head()
#Distribution for binary class variable 
print(data['normality'].value_counts())
sns.countplot(x="normality",data=data)
#Visualize heatmap of correlations
corr_mtx = data.corr()
fig, ax = plt.subplots()
sns.heatmap(corr_mtx, square=True)
#Pair plot showing pairwise relationships between features that are somewhat correlated from heatmap 
use_cols = ['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle','sacral_slope','pelvic_radius',\
            'degree_spondylolisthesis','normality']
sns.pairplot(data[use_cols],size=2,hue='normality')
plt.show()
#Convert categorical class to numerical binary
X = data.copy()
X = pd.get_dummies(X)
X.drop(X.columns[-1], axis=1, inplace=True)
X.rename(columns = {X.columns[-1]:'normality'},inplace=True)
X.head()
#Split train/test test
X_train,X_test,y_train,y_test = train_test_split(X.drop(X.columns[-1], axis=1),X['normality'],test_size=0.4,random_state=0)
print('X_train: ',X_train.shape)
print('X_test: ',X_test.shape)
print('y_train: ',y_train.shape)
print('y_test: ',y_test.shape)

fig, ax =plt.subplots(1,2)
sns.countplot(y_train,ax=ax[0])
ax[0].set_title('y_train')
sns.countplot(y_test,ax=ax[1])
ax[1].set_title('y_test')
#Fit Nearest Neighbor model
model = KNeighborsClassifier(n_neighbors=3)
clf = model.fit(X_train,y_train)

train_score = clf.score(X_train, y_train)
test_score  = clf.score(X_test, y_test)

print("Nearest Neighbor Model: ")
print ("Training Score: {}\nTest Score: {}" .format(train_score, test_score))
#Fit Naive Bayes model
model = GaussianNB()
clf = model.fit(X_train,y_train)

train_score = clf.score(X_train, y_train)
test_score  = clf.score(X_test, y_test)

print("Naive Bayes Model: ")
print ("Training Score: {}\nTest Score: {}" .format(train_score, test_score))
#Fit logistic regression model 
model = LogisticRegression(random_state=0)
clf = model.fit(X_train,y_train)

train_score = clf.score(X_train, y_train)
test_score  = clf.score(X_test, y_test)

print("Logistic Regression Model: ")
print ("Training Score: {}\nTest Score: {}" .format(train_score, test_score))
#Fit SVM model - does not do as good probably bc SVM are better when there are alot of features 
model = LinearSVC(random_state=0)
clf = model.fit(X_train,y_train)

train_score = clf.score(X_train, y_train)
test_score  = clf.score(X_test, y_test)

print("SVM Model:")
print ("Training Score: {}\nTest Score: {}" .format(train_score, test_score))
#Fit Random Forest Model
model = RandomForestRegressor(max_depth=5,n_estimators=30,random_state=0)
clf = model.fit(X_train,y_train)

train_score = clf.score(X_train, y_train)
test_score  = clf.score(X_test, y_test)

print("Random Forest Model:")
print ("Training Score: {}\nTest Score: {}" .format(train_score, test_score))