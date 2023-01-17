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
from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()

cancer.keys()

#This is a special type of dataset of sklearn
print(cancer["DESCR"])
cancer["data"]

# This is the data
cancer["feature_names"]

#This is the feature names of the data
df=pd.DataFrame(cancer["data"],columns=cancer["feature_names"])

df.head()
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(df)

scaled_data=scaler.transform(df)
scaled_data 
from sklearn.decomposition import PCA

pca=PCA(n_components=2) # we make an instance of PCA and decide how many components we want to have

pca.fit(scaled_data) # We make PCA fit to our scaled data
transformed_data=pca.transform(scaled_data)
scaled_data.shape

#This is the original shape of the data with 569 rows and 30 columns
transformed_data.shape

#Here we see 569 rows but 2 columns or components after PCA implementation
transformed_data
plt.figure(figsize=(15,10))

plt.scatter(transformed_data[:,0],transformed_data[:,1])

plt.xlabel("The First Principal Component")

plt.ylabel("The Second Principal Component")

#Here we plot all the rows of columns 1 and column 2 in a scatterplot.
plt.figure(figsize=(15,10))

plt.scatter(transformed_data[:,0],transformed_data[:,1],c=cancer["target"],cmap="plasma")

plt.xlabel("The First Principal Component")

plt.ylabel("The Second Principal Component")

#Here we plot all the rows of columns 1 and column 2 in a scatterplot.
pca.components_
df_comp=pd.DataFrame(pca.components_,columns=cancer["feature_names"])

df_comp
plt.figure(figsize=(15,10))

sns.heatmap(df_comp,cmap="magma")
X=transformed_data

y=cancer["target"]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LogisticRegression

log_regression=LogisticRegression()

log_regression.fit(X_train,y_train)
#Now our model is ready to predict the test data

predictions=log_regression.predict(X_test)

predictions
#Now it is time to evaluate how good the predictions are

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))

#The precision and accuracy precentages are over %90, it is very good
from sklearn.svm import SVC

svm_model=SVC()

svm_model.fit(X_train,y_train)
predictions=svm_model.predict(X_test)

predictions
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions))

#here we get the classification report to learn how accurate our model is