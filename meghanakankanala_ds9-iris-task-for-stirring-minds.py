# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#importing all the required libraries
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data= pd.read_csv("/kaggle/input/iris/Iris.csv")
print(data.shape)
data.head()
#getting to understand the data distribution 
data.describe()
#to check the categorical data distribution
data.describe(include="O")
#checking for null/empty data values
data.isnull().sum()
data.Species.unique()
fig,ax=plt.subplots(nrows=1,ncols=4,figsize=(20,20))
fig.suptitle("Relation of all the individual features with species",fontsize="20")
sns.barplot(data=data,x="Species",y="SepalLengthCm",ax=ax[0])
sns.barplot(data=data,x="Species",y="SepalWidthCm",ax=ax[1])
sns.barplot(data=data,x="Species",y="PetalLengthCm",ax=ax[2])
sns.barplot(data=data,x="Species",y="PetalWidthCm",ax=ax[3])
plt.show()
#dropping id column as it is unique for every data point and is of no use
data=data.drop("Id",axis=1,inplace=False)
#now ploting the pair graph which gives the graphs between all features
sns.pairplot(data=data)
plt.show()
#now checking the correlation between the features and plotting heatmap
correlation_val=data.corr()
sns.heatmap(correlation_val,annot=True)
#converting the categorical species column into numerical using label encoder
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data["Species"]=encoder.fit_transform(data["Species"])
data["Species"].unique()


#now splitting the data into x&y variables
y=data["Species"]
x=data.drop("Species",axis=1,inplace=False)
#(b)
#splitting data into train,test splits
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=30,random_state=1)
#(c)
#importing naivebayes model and training the model on train data set
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x_train,y_train)

#(d)
#predicting for test data set
pred=model.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred))


#here are the predicted values of model on test data set,
print("prediction list on test data which is 30% of the data in encoded numbers:",pred)
print("prediction list on test data which is 30% of the data : \n" , encoder.inverse_transform(pred))
print()
