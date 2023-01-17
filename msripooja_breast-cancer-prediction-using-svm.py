#for loading data and for performing data analysis operations on it

import pandas as pd

import numpy as np



#for data visualization

import seaborn as sns 

import matplotlib.pyplot as plt



#for PCA (feature engineering)

from sklearn.decomposition import PCA



#for data scaling

from sklearn.preprocessing import StandardScaler



#for splitting dataset

from sklearn.model_selection import train_test_split



#for fitting SVM model

from sklearn.svm import SVC



#for displaying evaluation metrics

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



#for file operations

import os



print("All required libraries loaded!")
#check the files in the given input folder

print(os.listdir("../input/"))
#load dataset into pandas dataframe

df = pd.read_csv("../input/data.csv")

df.shape
#check the data types of all the attributes loaded into the dataframe

df.dtypes
#see first few rows of the data loaded

df.head()
#see last few rows of the data loaded

df.tail()
#loading the predictors into dataframe 'X'

#NOTE: we are not choosing columns - 'id', 'diagnosis', 'Unnamed:32'

X = df.iloc[:,2:32]

print(X.shape)

X.head()
#loading target values into dataframe 'y'

y = df.diagnosis

print(y.shape)

y.head()
#coverting categorical data to numerical data

y_num = pd.get_dummies(y)

y_num.tail()
#use only one column for target value

y = y_num.M

print(y.shape)

y.tail()
#call corr() on dataframe X

X.corr()
plt.figure(figsize=(18, 12))

sns.heatmap(X.corr(), vmin=0.85, vmax=1, annot=True, cmap='YlGnBu', linewidths=.5)
#reducing the attributes in X dataframe



#1 scale the data

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



#2 drop the highly correlated columns which are not useful i.e., area, perimeter, perimeter_worst, area_worst, perimeter_se, area_se 

X_scaled = pd.DataFrame(X_scaled)

X_scaled_drop = X_scaled.drop(X_scaled.columns[[2, 3, 12, 13, 22, 23]], axis=1)



#3 apply PCA on scaled data

pca = PCA(n_components=0.95)

x_pca = pca.fit_transform(X_scaled_drop)

x_pca = pd.DataFrame(x_pca)



print("Before PCA, X dataframe shape = ",X.shape,"\nAfter PCA, x_pca dataframe shape = ",x_pca.shape)
print(pca.explained_variance_ratio_) 

print(pca.explained_variance_ratio_.sum())
#combine PCA data and target data



#1 set column names for the dataframe

colnames = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','diagnosis']



#target data

diag = df.iloc[:,1:2]



#combine PCA and target data

Xy = pd.DataFrame(np.hstack([x_pca,diag.values]),columns=colnames)



Xy.head()
#visualize data

sns.lmplot("PC1", "PC2", hue="diagnosis", data=Xy, fit_reg=False, markers=["o", "x"])

plt.show()
X=(Xy.iloc[:,0:11]).values

#75:25 train:test data splitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)



print("X_train shape ",X_train.shape)

print("y_train shape ",y_train.shape)

print("X_test shape ",X_test.shape)

print("y_test shape ",y_test.shape)
#model fitting

svc = SVC()

svc.fit(X_train, y_train)
#predict values

y_pred_svc =svc.predict(X_test)

y_pred_svc.shape
#print confusion matrix

cm = confusion_matrix(y_test, y_pred_svc)

print("Confusion matrix:\n",cm)
#print classification report

creport = classification_report(y_test, y_pred_svc)

print("Classification report:\n",creport)