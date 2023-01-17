# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_wine_data = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df_wine_data.shape
df_wine_data.columns
df_wine_data.dtypes
for col in df_wine_data.columns :

    print(df_wine_data[col].isnull().value_counts())
fig = plt.figure(figsize = (8,5))

ax = fig.add_subplot(111)

df_wine_data['quality'].plot(kind = 'hist',bins=20,ax=ax)

ax.set_xlabel('Quality',size = 12)
#df_wine_data.loc[df_wine_data['quality']>=6.5,'quality'] =1

#df_wine_data.loc[(df_wine_data['quality']<6.5) & (df_wine_data['quality'] !=1) ,'quality'] =0
#GoodOrBad feature will used for classification in later part

df_wine_data['GoodOrBad'] = df_wine_data['quality']

df_wine_data.loc[df_wine_data['GoodOrBad']>=6.5,'GoodOrBad'] =1

df_wine_data.loc[(df_wine_data['GoodOrBad']<6.5) & (df_wine_data['GoodOrBad'] !=1) ,'GoodOrBad'] =0
#Fixed Acidity have a similar kind of level in all kinds of wines 

sns.barplot(data= df_wine_data,x='quality',y='fixed acidity')
#Sugar level are pretty much same irrespective of quality.

sns.barplot(data= df_wine_data,x='quality',y='residual sugar')
#Cholrides level decrease as quality increases

sns.barplot(data= df_wine_data,x='quality',y='chlorides')
#citric acid level increases as quality increases

sns.barplot(data= df_wine_data,x='quality',y='citric acid')
#volatile acidity level decrease as quality increases

sns.barplot(data= df_wine_data,x='quality',y='volatile acidity')
#free sulfur dioxide are high in midium quality of wines and low in low quality of wines.

sns.barplot(data= df_wine_data,x='quality',y='free sulfur dioxide')
#like free sulfur dioxide, total sulfur dioxide are also high in midium quality of wines and low in low quality of wines.

sns.barplot(data= df_wine_data,x='quality',y='total sulfur dioxide')
#Bar plot is unable to give any significant info lets try a different plot 

sns.barplot(data= df_wine_data,x='quality',y='density')
#density is pretty is in similar levels irrespective of quality

sns.swarmplot(data= df_wine_data,x='quality',y='density')



sns.barplot(data= df_wine_data,x='quality',y='pH')
#Majority Wines with quality 5,6,7 tend have a pH value between 3.6 to 3.0

#but wines with quality 3,8 also same kind of range 3.6 to 3.2

#pH level will not be able to play a significant role in deciding the quality

sns.swarmplot(data= df_wine_data,x='quality',y='pH')
#sulphates quantity is showing upward trend w.r.t to quality

sns.barplot(data= df_wine_data,x='quality',y='sulphates')
#Alcohol content is high in wines with quality 7,8 but similar in wines

#with quality 3,4,5,6

sns.barplot(data= df_wine_data,x='quality',y='alcohol')
columns = ['volatile acidity', 'citric acid',

       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates', 'alcohol']
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

std_scaler.fit(df_wine_data[columns])

df_wine_data[columns] = std_scaler.transform(df_wine_data[columns])
df_wine_data.shape
#GoodOrBad is just a categorical variable that we have created earlier

#This plot shows us the strength of coorelation of feature with quality variable.

df_wine_data.corr()['quality'].plot(kind = 'bar')
#Through the coorelation matrix we can see that 'volatile acidity', 'citric acid',

#'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates', 'alcohol'

#features have strong coorelation with quality

fig = plt.figure(figsize = (12,6))

sns.heatmap(df_wine_data.corr(),annot = True)
df_wine_data.columns
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(np.array(df_wine_data[columns])

                                                    ,np.array(df_wine_data['quality']),random_state =20,test_size =0.4,train_size = 0.6)
X_val, X_test, y_val, y_test = train_test_split(X_test,y_test,random_state =20,test_size =0.5,train_size = 0.5)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures

mses= []

ks = np.arange(1,5,1)

for k in ks :

    poly = PolynomialFeatures(k)

    X_train_poly = poly.fit_transform(X_train)

    X_val_poly = poly.fit_transform(X_val)

    reg = LinearRegression()

    reg.fit(X_train_poly,y_train)

    y_hat = reg.predict(X_val_poly)

    mses.append(mean_squared_error(y_val,y_hat))
plt.plot(ks,mses)

plt.xticks(ks)

plt.xlabel('degree of polynomial',size =12)

plt.ylabel('MSE',size =12)
#We will use polynomial feature with degree 3 because if we increase degree further

#MSE incresing which we don't want

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import Lasso

ks = np.arange(1,10,1) *0.01

mses = []

for k in ks :

    poly = PolynomialFeatures(3)

    X_train_poly = poly.fit_transform(X_train)

    X_val_poly = poly.fit_transform(X_val)

    reg = Lasso(alpha = k,max_iter =1000)

    reg.fit(X_train_poly,y_train)

    y_hat = reg.predict(X_val_poly)

    mses.append(mean_squared_error(y_val,y_hat))
plt.figure(figsize=(12,5))

plt.plot(ks,mses)

plt.xticks(ks)

plt.xlabel('Regularization Parameter',size =12)

plt.ylabel('MSE on Validation Dataset',size =12)

plt.gca().ticklabel_format(axis='both', style='plain', useOffset=False)
#We consider regularization parameter as 0.02

poly = PolynomialFeatures(3)

X_train_poly = poly.fit_transform(X_train)

X_val_poly = poly.fit_transform(X_val)

reg = Lasso(alpha = 0.02)

mses_train = []

mses_val = []

split_arr = np.arange(179,717,179)

temp = 0

for i in split_arr :

    reg.fit(X_train_poly[temp:i,:],y_train[temp:i])

    y_train_hat = reg.predict(X_train_poly)

    y_val_hat = reg.predict(X_val_poly)

    mses_train.append(mean_squared_error(y_train,y_train_hat))

    mses_val.append(mean_squared_error(y_val,y_val_hat))

    i =i+1

    temp =1
plt.figure(figsize = (13,4))

plt.plot(split_arr,mses_train,label = 'Training Error')

plt.plot(split_arr,mses_val,label = 'Validation Error')

plt.xlabel('Training Sample',size =12)

plt.ylabel('MSE',size =12)

plt.legend()

plt.gca().ticklabel_format(axis='both', style='plain', useOffset=False)
#with best degree of polynomial as and regularization paramter

#will try to predict on test data

poly = PolynomialFeatures(3)

X_test_poly = poly.fit_transform(X_test)

y_test_hat = reg.predict(X_test_poly)

mean_squared_error(y_test,y_test_hat)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(np.array(df_wine_data[columns])

                                                    ,np.array(df_wine_data['GoodOrBad']),random_state =20,test_size =0.4,train_size = 0.6)
X_val, X_test, y_val, y_test = train_test_split(X_test,y_test,random_state =20,test_size =0.5,train_size = 0.5)
from sklearn.svm import SVC

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

clf = SVC()

clf.fit(X_train,y_train)

y_val_hat = clf.predict(X_val)
#Performance on Validation Set

print('Accuracy Score :',accuracy_score(y_val,y_val_hat))

print('Precision Score :',precision_score(y_val,y_val_hat))

print('Recall Score :',recall_score(y_val,y_val_hat))

sns.heatmap(confusion_matrix(y_val,y_val_hat),annot =True,fmt='d')

#Performance of Test Set

y_tst_hat = clf.predict(X_test)

print('Accuracy Score :',accuracy_score(y_test,y_tst_hat))

print('Precision Score :',precision_score(y_test,y_tst_hat))

print('Recall Score :',recall_score(y_test,y_tst_hat))

sns.heatmap(confusion_matrix(y_test,y_tst_hat),annot =True,fmt='d')
#figure = plt.figure(figsize = (10,10))

#ax = figure.add_subplot(111)

#sns.boxplot(data = df_wine_data[columns],width = 0.8,orient = 'h')

#Here we are using 1.5 times Inter Quartile Range to filter outliers there are other ways also like Z-Score. 

#Q1 = df_wine_data[columns].quantile(0.25) #First Quartile

#Q3 = df_wine_data[columns].quantile(0.75) #Thrid Quartile

#IQR = Q3 - Q1                             #Inter Quartile Range

#df_wine_data = df_wine_data[~((df_wine_data < (Q1 - 1.5 * IQR)) |(df_wine_data > (Q3 + 1.5 * IQR))).any(axis=1)]