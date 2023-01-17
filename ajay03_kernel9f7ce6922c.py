# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from random import randrange, uniform

from scipy.stats import chi2_contingency

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
marketing_train = pd.read_csv("/kaggle/input/wine-quality/winequalityN.csv")
marketing_train.head(10)
marketing_train.shape
#Create dataframe with missing percentage

missing_val = pd.DataFrame(marketing_train.isnull().sum())



#Reset index

missing_val = missing_val.reset_index()



#Rename variable

missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})



#Calculate percentage

missing_val['Missing_percentage'] = (missing_val['Missing_percentage']/len(marketing_train))*100



#descending order

missing_val = missing_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)

missing_val
var = ['pH','volatile acidity','residual sugar']

for i in var:

    marketing_train[i] = marketing_train[i].fillna(marketing_train[i].mean())

VAR = ['fixed acidity','sulphates','citric acid','chlorides']

for i in VAR:

    marketing_train[i] = marketing_train[i].fillna(marketing_train[i].median())

marketing_train.isnull().sum()
# #Plot boxplot to visualize Outliers

%matplotlib inline  

plt.boxplot(marketing_train['fixed acidity'])
# #Plot boxplot to visualize Outliers

%matplotlib inline  

plt.boxplot(marketing_train['citric acid'])
# #Plot boxplot to visualize Outliers

%matplotlib inline  

plt.boxplot(marketing_train['volatile acidity'])
# #Plot boxplot to visualize Outliers

%matplotlib inline  

plt.boxplot(marketing_train['residual sugar'])
# #Plot boxplot to visualize Outliers

%matplotlib inline  

plt.boxplot(marketing_train['chlorides'])
# #Plot boxplot to visualize Outliers

%matplotlib inline  

plt.boxplot(marketing_train['free sulfur dioxide'])
# #Plot boxplot to visualize Outliers

%matplotlib inline  

plt.boxplot(marketing_train['total sulfur dioxide'])
# #Plot boxplot to visualize Outliers

%matplotlib inline  

plt.boxplot(marketing_train['density'])
# #Plot boxplot to visualize Outliers

%matplotlib inline  

plt.boxplot(marketing_train['pH'])
# #Plot boxplot to visualize Outliers

%matplotlib inline  

plt.boxplot(marketing_train['sulphates'])
# #Plot boxplot to visualize Outliers

%matplotlib inline  

plt.boxplot(marketing_train['alcohol'])
# #Plot boxplot to visualize Outliers

%matplotlib inline  

plt.boxplot(marketing_train['quality'])
#save numeric names

cnames =  ['fixed acidity', 'volatile acidity', 'citric acid',

       'residual sugar', 'chlorides', 'free sulfur dioxide',

       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',

       'quality']
# #Detect and delete outliers from data

for i in cnames:

     print(i)

     q75, q25 = np.percentile(marketing_train.loc[:,i], [75 ,25])

     iqr = q75 - q25



     min = q25 - (iqr*1.5)

     max = q75 + (iqr*1.5)

     print(min)

     print(max)

    

     marketing_train = marketing_train.drop(marketing_train[marketing_train.loc[:,i] < min].index)

     marketing_train = marketing_train.drop(marketing_train[marketing_train.loc[:,i] > max].index)
marketing_train.shape
##Correlation analysis

#Correlation plot

df_corr = marketing_train.loc[:,cnames]
#Set the width and hieght of the plot

f, ax = plt.subplots(figsize=(7, 5))



#Generate correlation matrix

corr = df_corr.corr()



#Plot using seaborn library

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)
marketing_train = marketing_train.drop(['density'], axis=1)
marketing_train = marketing_train.drop(['alcohol'], axis=1)
marketing_train.shape
names =  ['fixed acidity', 'volatile acidity', 'citric acid',

       'residual sugar', 'chlorides', 'free sulfur dioxide',

       'total sulfur dioxide','pH', 'sulphates',

       'quality']
#Normality check

for i in names:

    %matplotlib inline  

    plt.hist(marketing_train[i], bins='auto')

    
# standredizaion for fixed acidity

marketing_train['fixed acidity'] = (marketing_train['fixed acidity']- 6.91)/0.83

# standradzation for citric acid

marketing_train['citric acid'] = (marketing_train['citric acid']- 0.32)/0.09
marketing_train['total sulfur dioxide'] = (marketing_train['total sulfur dioxide']- 130.33)/47.03
# normalization for volatile acidity

marketing_train['volatile acidity'] = (marketing_train['volatile acidity']- 0.08)/0.56
marketing_train['residual sugar'] = (marketing_train['residual sugar']- 0.6)/18.35
marketing_train['chlorides'] = (marketing_train['chlorides']- 0.009)/.072
marketing_train['free sulfur dioxide'] = (marketing_train['free sulfur dioxide']- 2)/76
marketing_train['pH'] = (marketing_train['pH']- 2.82)/0.77
marketing_train['sulphates'] = (marketing_train['sulphates']- 0.22)/0.57
marketing_train['quality'] = (marketing_train['quality']- 4)/3
marketing_train.head(10)
from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
#replace target categories with Yes or No

marketing_train['type'] = marketing_train['type'].replace(0, 'red')

marketing_train['type'] = marketing_train['type'].replace(1, 'white')
#Divide data into train and test

X = marketing_train.values[:, 1:10]

Y = marketing_train.values[:,0]



X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2)
from sklearn.ensemble import RandomForestClassifier



RF_model = RandomForestClassifier(n_estimators = 10).fit(X_train, y_train)
RF_Predictions = RF_model.predict(X_test)
#build confusion matrix

# from sklearn.metrics import confusion_matrix 

# CM = confusion_matrix(y_test, y_pred)

CM = pd.crosstab(y_test, RF_Predictions)



#let us save TP, TN, FP, FN

TN = CM.iloc[0,0]

FN = CM.iloc[1,0]

TP = CM.iloc[1,1]

FP = CM.iloc[0,1]



#check accuracy of model

#accuracy_score(y_test, y_pred)*100

((TP+TN)*100)/(TP+TN+FP+FN)



#False Negative rate 

#(FN/(FN+TP))*100



#Accuracy:  98.98

#FNR:  0.87