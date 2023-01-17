# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/BlackFriday.csv")
df.info()
df.head()
df.describe()
df.tail()
from sklearn import neighbors
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
sns.set()
#Identify numerical columns to produce a heatmap
catcols = ['User_ID','Product_ID','Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3','Purchase']
numcols = [x for x in df.columns if x in catcols]
print( numcols )
#Lets start by plotting a heatmap to determine if any variables are correlated
plt.figure(figsize = (12,8))
sns.heatmap(data=df[numcols].corr())
plt.show()
plt.gcf().clear()
missing_df = df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.loc[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')

ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(6,6))
rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()
product_Category_3_median = df.loc[df['Product_Category_3'] > 0, 'Product_Category_3'].median()
print(product_Category_3_median)

product_Category_2_median = df.loc[df['Product_Category_2'] > 0, 'Product_Category_2'].median()
print(product_Category_2_median)
index = df.Product_Category_3.isnull()
df.loc[index,'Product_Category_3'] = product_Category_3_median
index1 = df.Product_Category_2.isnull()
df.loc[index1,'Product_Category_2'] = product_Category_2_median
df.head()
df.info()
corr_val=df.corr(method='pearson')
print(corr_val)
df['Male'] = df['Gender'].map( {'M':1, 'F':0} )
df[['Gender', 'Male']].head()
df = df.drop(['Gender'], axis=1)
df.head()
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(df['City_Category'])
encoder.classes_
df['City_encoded'] = encoder.transform(df['City_Category']) # transform as a separate step from fit
df[['City_Category', 'City_encoded']].head()
df = df.drop(['City_Category'], axis=1)
df.head()
encoder = LabelEncoder()
encoder.fit(df['Age'])
encoder.classes_
df['Age_encoded'] = encoder.transform(df['Age']) # transform as a separate step from fit
df[['Age', 'Age_encoded']].head()
df = df.drop(['Age'], axis=1)
df.head()
encoder = LabelEncoder()
encoder.fit(df['Stay_In_Current_City_Years'])
encoder.classes_
index = df.Stay_In_Current_City_Years == '4+'
df.loc[index,'Stay_In_Current_City_Years'] = 4
df.head()
catcols = ['User_ID','Product_ID','Occupation','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3','Purchase','Male','City_encoded','Age_encoded']
numcols = [x for x in df.columns if x in catcols]
plt.figure(figsize = (12,8))
sns.heatmap(data=df[numcols].corr())
plt.show()
plt.gcf().clear()
df.describe()
corr_val=df.corr(method='pearson')
print(corr_val)
corr_val=df.corr(method='pearson')
print(corr_val)
# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling
df['Purchase_update'] = df['Purchase']
df = df.drop(['Purchase'],axis=1)
df.head()

encoder = LabelEncoder()
encoder.fit(df['Product_ID'])
encoder.classes_
df.Product_ID = df.Product_ID.str.replace('P','')
df.head()
df.info()
from sklearn.preprocessing import Normalizer
ac_data=df.values
X=ac_data[:,0:10]
Y=ac_data[:,10]
scaler=Normalizer()
rescaledx=scaler.fit_transform(X)
print(rescaledx)
df.info()
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
print(rescaledx)
models=[]
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('SVM',SVC()))
models.append(('NB',GaussianNB()))
models.append(('KNN',KNeighborsClassifier()))



