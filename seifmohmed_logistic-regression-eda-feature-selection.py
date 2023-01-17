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

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from pandas_profiling import ProfileReport
candy = pd.read_csv('../input/the-ultimate-halloween-candy-power-ranking/candy-data.csv')
candy.head()
candy.shape
candy.info()
def count(feature):

    

    # Show the counts of observations in each categorical bin using bars

    sns.countplot(x=feature,data=candy)

    
candy.head()
fig, ax = plt.subplots(3, 3,figsize=(15,20))

plt.subplot(3,3,1)

count('chocolate')

plt.subplot(3,3,2)

count('fruity')

plt.subplot(3,3,3)

count('caramel')

plt.subplot(3,3,4)

count('peanutyalmondy')

plt.subplot(3,3,5)

count('nougat')

plt.subplot(3,3,6)

count('crispedricewafer')

plt.subplot(3,3,7)

count('bar')

plt.subplot(3,3,8)

count('pluribus')

plt.subplot(3,3,9)

count('hard')
def box(var):

    # this function take the variable and return a boxplot for each type of fish

    sns.boxplot(x="chocolate", y=var, data=candy,palette='rainbow')
fig, ax = plt.subplots(3, 1,figsize=(15,20))

plt.subplot(3,1,1)

box('sugarpercent')

plt.subplot(3,1,2)

box('pricepercent')

plt.subplot(3,1,3)

box('winpercent')
candy.head()
X = candy.drop(['chocolate','competitorname'],axis=1) #independent columns

y = candy['chocolate']   #target column i.e price range

from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model = ExtraTreesClassifier()

model.fit(X,y)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

plt.figure(figsize=(10,10))

feat_importances.nlargest(8).plot(kind='barh')

plt.show()
plt.figure(figsize=(15,10))

sns.heatmap(candy.corr(),cmap='coolwarm',annot=True,linecolor='white',linewidths=4)
candy.info()
competitorname = pd.get_dummies(candy['competitorname'],drop_first=True)
candy.drop('competitorname',axis=1,inplace=True)
candy=pd.concat([candy,competitorname],axis=1)
candy.head()
X = candy.drop('chocolate',axis=1)

y = candy['chocolate']
#spliting the dataset into training set and test set

from sklearn.model_selection import train_test_split

X_train ,X_test , y_train , y_test =train_test_split(X,y, test_size = 0.2 , random_state=4)
from sklearn.linear_model import LogisticRegression

log = LogisticRegression()
log.fit(X_train,y_train)
predictions = log.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
confusion_matrix(y_test,predictions)
from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, predictions))