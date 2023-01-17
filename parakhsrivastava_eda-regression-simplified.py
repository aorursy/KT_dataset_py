import numpy as np

# Scientific computing library

import pandas as pd 

# DataFrame dealing Library

import seaborn as sns

# Graphical/Plotting Library

import matplotlib.pyplot as plt

# Graphical/Plotting Library

%matplotlib inline
data=pd.read_csv("../input/Admission_Predict_Ver1.1.csv")

data.head()

# First 5 rows of the dataset
new_data=data.drop('Serial No.',axis=1)

# above one is 1st way

# new_data=data.iloc[:,1:9] <-- Skip 1st column, this is 2nd way

new_data.head()
new_data.describe()

# You can see "mean" of everything and that will be shown in countplot afterwards too!
new_data.isnull().sum(axis=0)
f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(new_data.corr(),annot=True)
plt.subplots(figsize=(20,4))

sns.barplot(x="GRE Score",y="Chance of Admit ",data=data)

plt.subplots(figsize=(25,5))

sns.barplot(x="TOEFL Score",y="Chance of Admit ",data=data)

plt.subplots(figsize=(20,4))

sns.barplot(x="University Rating",y="Chance of Admit ",data=data)

plt.subplots(figsize=(15,5))

sns.barplot(x="SOP",y="Chance of Admit ",data=data)
temp_series = new_data.Research.value_counts()

# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.value_counts.html

labels = (np.array(temp_series.index))

# https://docs.scipy.org/doc/numpy-1.15.0/user/basics.creation.html

# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.index.html

sizes = (np.array((temp_series/temp_series.sum())*100))

# calculating %ages

colors = ['Pink','SkyBlue']

plt.pie(sizes,labels = labels,colors=colors,autopct='%1.1f%%',shadow=True,startangle=90)

# https://www.commonlounge.com/discussion/9d6aac569e274dacbf90ed61534c076b#pie-chart

# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pie.html

plt.title("Research Percentage")

plt.show()
plt.subplots(figsize=(20,4))

sns.countplot(x="GRE Score",data=data)

plt.subplots(figsize=(25,5))

sns.countplot(x="TOEFL Score",data=data)

plt.subplots(figsize=(20,4))

sns.countplot(x="University Rating",data=data)

plt.subplots(figsize=(15,5))

sns.countplot(x="SOP",data=data)
X=new_data.iloc[:,:7]

y=new_data["Chance of Admit "]
print(X.shape)

print(y.shape)

X.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=50)
from sklearn.linear_model import LinearRegression

#Linear Regression

regressor=LinearRegression()

regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

y_pred
from sklearn.metrics import mean_absolute_error,r2_score

print("R2 score ",r2_score(y_pred,y_test))

print("mean_absolute_error ",mean_absolute_error(y_pred,y_test))