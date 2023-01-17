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
dataset=pd.read_csv('/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv')
dataset.head()
dataset.shape

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
sns.countplot(dataset['target_class'],label='count')
dataset.hist(bins=10,figsize=(20,15))

plt.show()
dataset.corr()
sns.pairplot(data=dataset,

             palette="husl",

             hue="target_class",

             vars=[" Mean of the integrated profile",

                   " Excess kurtosis of the integrated profile",

                   " Skewness of the integrated profile",

                   " Mean of the DM-SNR curve",

                   " Excess kurtosis of the DM-SNR curve",

                   " Skewness of the DM-SNR curve"])



plt.suptitle("PairPlot of Data Without Std. Dev. Fields",fontsize=18)



plt.tight_layout()

plt.show()
plt.figure(figsize=(20,16))

sns.heatmap(data=dataset.corr(),annot=True)

plt.title('Co-Relation Mattrix')

plt.tight_layout()

plt.show()

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics 

from sklearn.metrics import r2_score,accuracy_score,confusion_matrix,classification_report
X=dataset.drop('target_class',axis=1)

y=dataset['target_class']
model=DecisionTreeClassifier()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model.fit(X_train,y_train)
pred=model.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
r2_score(y_test,pred)
accuracy_score(y_test,pred)*100
df=pd.DataFrame({'Actual Pred':y_test,'Predicted ':pred})

df1=df.head(25)

print(df1)
df1.plot(kind='bar',figsize=(20,5))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
print('Mean absolute Error',metrics.mean_absolute_error(y_test,pred))

print('Mean squared Error',metrics.mean_squared_error(y_test,pred))

print('Mean squared Error',np.sqrt(metrics.mean_absolute_error(y_test,pred)))
from sklearn.linear_model import LogisticRegression
model2=LogisticRegression()

model2.fit(X_train,y_train)

y_pred=model2.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
accuracy_score(y_test,y_pred)*100
from sklearn.linear_model import LinearRegression
model3=LinearRegression()

model3.fit(X_train,y_train)
lin_pred=model3.predict(X_test)
r2_score(y_test,lin_pred)
from sklearn.svm import LinearSVC
model3=LinearSVC(C=1000)
model3.fit(X_train,y_train)

prediction=model3.predict(X_test)
print(confusion_matrix(y_test,prediction))
print(accuracy_score(y_test,prediction)*100)
print(classification_report(y_test,prediction))