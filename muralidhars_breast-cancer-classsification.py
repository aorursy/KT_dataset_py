# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Any results you write to the current directory are saved as output.
#importing dataset



from sklearn.datasets import load_breast_cancer

cancer= load_breast_cancer()
cancer
cancer.keys()
cancer['data'].shape
cancer['target']
print(cancer['target_names'])
print(cancer['DESCR'])
print(cancer['feature_names'])
#creating a dataframe

df_cancer=pd.DataFrame(data=np.c_[cancer['data'],cancer['target']],columns=np.append(cancer['feature_names'],['target']))
df_cancer.head()
df_cancer.tail()
#visualizing the correlation between features using plots

# Using hue helps us to classify the datapoints based on the dependent varaiable

sns.pairplot(df_cancer,hue='target', vars=['mean radius','mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity'])
#Lets use countplot to number of samples each target category holds.

sns.countplot(df_cancer['target'])
sns.heatmap(df_cancer.corr(),annot=True)
#resizing the map size for clear visualization

plt.figure(figsize=(20,20))

sns.heatmap(df_cancer.corr(),annot=True)
#Training the model

x=df_cancer.drop(['target'],axis=1)

y=df_cancer['target']
x.head()
y.head(3)
#check for missing values

x.isnull().sum()
y.isnull().sum()
# without feature scaling



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size= 0.25, random_state=10)
x_train.shape

y_train.shape

x_test.shape

y_test.shape
y_train.shape
y_train.head()
x_test.shape

y_test.shape
# fitting SVC 

from sklearn.svm import SVC

classifier= SVC()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
# analyzing accuracy using metrics

from sklearn.metrics import confusion_matrix, classification_report



matrix= confusion_matrix(y_test,y_pred)

sns.heatmap(matrix, annot=True)

print(classification_report(y_test,y_pred))
# data normalization using feature Scaling

from sklearn.preprocessing import StandardScaler

sc_train=StandardScaler()

sc_test=StandardScaler()

x_train_scaled=sc_train.fit_transform(x_train)

x_test_scaled=sc_test.fit_transform(x_test)
classifier.fit(x_train_scaled,y_train)
y_pred_scaled=classifier.predict(x_test_scaled)

matrix= confusion_matrix(y_test,y_pred_scaled)

sns.heatmap(matrix, annot=True)
print(classification_report(y_test,y_pred_scaled))
grid_params=({'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001],'kernel':['rbf']})

from sklearn.model_selection import GridSearchCV



grid= GridSearchCV(SVC(), grid_params, refit=True, verbose=4)



grid.fit(x_train_scaled,y_train)
grid.best_params_
grid_pred = grid.predict(x_test_scaled)

matrix= confusion_matrix(y_test,grid_pred)

sns.heatmap(matrix, annot=True)
print(classification_report(y_test,grid_pred))