# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
iris_df=pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv")
iris_df.head()
print("Shape of the data frame: ",iris_df.shape)
print("Total null values: ",iris_df.isna().sum().sum())
print("Duplicate values: ",iris_df.duplicated().sum() )


iris_df.drop_duplicates(inplace=True)
print("Shape of the data frame: ",iris_df.shape)
print("\n")
print("Species categories with its count \n",iris_df["species"].value_counts())
iris_df.describe()
#iris_df.plot(kind='box')
#plt.show()
sns.boxplot(data=iris_df)
skewness_value=iris_df.skew(axis=0)
print("Measure of skewness column wise:\n",skewness_value)  

kurtosis_values=iris_df.kurt(axis=0)
print(kurtosis_values)

from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
iris_df["species"]=label_encoder.fit_transform(iris_df["species"])
print(iris_df.head(10))
print("\n")
print(iris_df["species"].value_counts())
X=iris_df.drop(["species"],axis=1)
Y=iris_df["species"]
from sklearn.model_selection import train_test_split
X_train1,X_test1,Y_train1,Y_test1=train_test_split(X,Y,test_size=0.2,random_state=3)
from sklearn.linear_model import LogisticRegression
logistic_reg=LogisticRegression()
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
logistic_reg.fit(X_train1,Y_train1)
predicted_result1=logistic_reg.predict(X_test1)
print("Accuracy Score: ",accuracy_score(Y_test1, predicted_result1))

print("Confusion Matrix:\n ",confusion_matrix(Y_test1, predicted_result1))

print("Classification Report:\n ",classification_report(Y_test1, predicted_result1))
from sklearn.model_selection import train_test_split
X_train2,X_test2,Y_train2,Y_test2=train_test_split(X,Y,test_size=0.2,random_state=5,stratify=Y)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train2=ss.fit_transform(X_train2)
X_test2=ss.transform(X_test2)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
LDA=LinearDiscriminantAnalysis()
X_train2=LDA.fit_transform(X_train2,Y_train2)
X_test2=LDA.transform(X_test2)
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(X_train2,Y_train2)
y_prediction=LR.predict(X_test2)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("\n Accuracy Score:",accuracy_score(Y_test2,y_prediction))
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test2,y_prediction))
print("Classification Report:")
print(classification_report(Y_test2,y_prediction))