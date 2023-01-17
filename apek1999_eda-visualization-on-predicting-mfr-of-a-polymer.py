import pandas as pd

import numpy as np

df=pd.read_csv("../input/algo.csv")

df.rename( columns={'Unnamed: 0':'date'}, inplace=True )

df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].dt.year

df['month'] = df['date'].dt.month

df['day'] = df['date'].dt.day

df['hour'] = df['date'].dt.hour

df['minute'] = df['date'].dt.minute

df=df.drop(["date"],axis=1)

df.head()
df.isnull().sum() # Checking if there is any null value in the dataset 
df.dtypes
df=df.fillna(df.mean()) #replacing the null values with the average value 
df.describe() #statistical description of the dataset
#Creating Pearson Correlation matrix 

import matplotlib.pyplot as plt #Importing matplotlib library

import seaborn as sns#importing seaborn

import statsmodels.api as sm

plt.figure(figsize=(10,8))

cor = df.corr()

sns.heatmap(cor, annot=True,cmap=plt.cm.Reds)

plt.show()
#Correlation with output variable

cor_tar = abs(cor["MFR"])

#Selecting highly correlated features with target value(MFR)

rel_features = cor_tar[cor_tar>0.01]

rel_features #Features those are highly correlated with target value('MFR')
import matplotlib.pyplot as plt 

# plot a histogram to see the distribution of MFR  

df['MFR'].hist(bins=30) 
# get columns to plot

columns = df.columns.drop(['MFR'])

# create x data

x_data = range(0,df.shape[0])

# create figure and axis

fig, ax = plt.subplots()

# plot each column

for column in columns:

    ax.plot(x_data, df[column], label=column)

# set title and legend

ax.set_title('Prediction of Polymer Quality')

ax.legend()
#Data Visulaization and relation between each and every feature(including the target value)

import seaborn as sns

sns.pairplot(df)
#fREQUENCY OF EACH AND EVERY FEATURES IN THE DAATSET

df.plot.hist(subplots=True, figsize=(10, 10), bins=20)
sns.countplot(df['MFR'])
sns.countplot(df['513HC31114-5.mv'])
sns.countplot(df['513PC31201.pv'])
sns.countplot(df['513FC31409.pv'])
# plotting between Hydrogen ratio and MFR value

y = df["MFR"] 

x = df["513HC31114-5.mv"] 

plt.scatter(x, y, label= "stars", color= "m",  

            marker= "*", s=30) 

# x-axis label 

plt.ylabel('MFR') 

# frequency label 

plt.xlabel('513HC31114-5.mv') 

# function to show the plot 

plt.show() 
# plotting between Pressure controller and MFR value 

y = df["MFR"] 

x = df["513PC31201.pv"] 

plt.scatter(x, y, label= "stars", color= "m",  

            marker= "*", s=30) 

# x-axis label 

plt.ylabel('MFR') 

# frequency label 

plt.xlabel('513PC31201.pv') 

# function to show the plot 

plt.show() 
# plotting between Propylene flow and MFR value

y = df["MFR"] 

x = df["513FC31103.pv"] 

plt.scatter(x, y, label= "stars", color= "m",  

            marker= "*", s=30) 

# x-axis label 

plt.ylabel('MFR') 

# frequency label 

plt.xlabel('513FC31103.pv') 

# function to show the plot 

plt.show() 
#Min-Max Scaling 

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

  

# Scaled feature 

df['513FC31103.pv'] = scaler.fit_transform(df['513FC31103.pv'].values.reshape(-1,1))

df['513PC31201.pv'] = scaler.fit_transform(df['513PC31201.pv'].values.reshape(-1,1))

df['513LC31202.pv'] = scaler.fit_transform(df['513LC31202.pv'].values.reshape(-1,1))

df['513FC31409.pv'] = scaler.fit_transform(df['513FC31409.pv'].values.reshape(-1,1))

df['513FC31103.pv'] = scaler.fit_transform(df['513FC31103.pv'].values.reshape(-1,1))

df['513TC31220.pv'] = scaler.fit_transform(df['513TC31220.pv'].values.reshape(-1,1))
df.head()
#Using Logistic Regression



#Split the dataset into train and test

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df.drop(["MFR","year","hour","minute"],axis=1),df.MFR.astype('int'),train_size=0.8) #TARGET VALUE = MFR 

#Building Model

from sklearn.linear_model import LogisticRegression

model=LogisticRegression(C=100)

model.fit(x_train,y_train)
#For Logistic Regression

from sklearn import metrics

y_pred=model.predict(x_test)

print("Model Accuracy is :",metrics.accuracy_score(y_test,y_pred))
#Using Random Forest



#Split the dataset into train and test

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df.drop(["MFR","year","hour","minute"],axis=1),df.MFR.astype('int'),train_size=0.8) #TARGET VALUE = MFR 

#Building Model

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=1000)

model.fit(x_train,y_train)
#For Random Forest

y_pred=model.predict(x_test)

print("Model Accuracy is :",metrics.accuracy_score(y_test,y_pred))
#Using KNN Algorithm



#Split the dataset into train and test

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df.drop(["MFR","year","hour","minute"],axis=1),df.MFR.astype('int'),train_size=0.8) #TARGET VALUE = MFR 

#Building Model

from sklearn.neighbors import KNeighborsClassifier

model=KNeighborsClassifier(n_neighbors=3)

model.fit(x_train,y_train)
#For KNN algorithm

y_pred=model.predict(x_test)

print("Model Accuracy is :",metrics.accuracy_score(y_test,y_pred))
#Using Decision Tree



#Split the dataset into train and test

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df.drop(["MFR","year","hour","minute"],axis=1),df.MFR.astype('int'),train_size=0.8) #TARGET VALUE = MFR 

#Building Model

from sklearn import tree

model=tree.DecisionTreeClassifier()

model.fit(x_train,y_train)
#For Decision Tree

y_pred=model.predict(x_test)

print("Model Accuracy is :",metrics.accuracy_score(y_test,y_pred))