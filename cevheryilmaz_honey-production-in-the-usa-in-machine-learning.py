####STEPS####
#1-import library
#2-insert dataset
#3-head()
#4-samplee()
#5-tail()
#6-shape
#7-describe()
#8-groupby()
#9-hist()
#10-corr()
#11-correlation map
#12-scatter Plot
#13-isnull()
#14-outliers
#15-create new attribute
#16-Normalization
#17-Mulitple Lineer Regression
#18-Naive Bayes
#19-model results and interpretation
#20-Interpration of results
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df = pd.read_csv('../input/honeyproduction.csv')       
df.info()      #memory usage and data types
df.head(10)        #first 10 rows
df.sample(6)     #random 6 rows 
df.tail(10)      #last 10 rows
df.shape          #dataset columns and lines 
df.describe()           #important values in the data(min,max,count,std,%)
df.groupby("year").size()             ## How are class distributions?
df.hist()         #sataset histogram review
df.corr()
#In the data set, there is a good relationship between the corr and the correct proportions and inverse proportions.
#Or: There is a directional link between stocks and totalprod.
#Or: There is a very ineffective connection between priceperib and yieldpercol.
#correlation map
f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
#The heat map showed that the color was open and there was a lateral connection.
# When the color is placed, there is a reverse connection.
#Scatter Plot
df.plot(kind='scatter', x='totalprod', y='prodvalue',alpha = 0.5,color = 'blue')
plt.xlabel('totalprod')              # label = name of label
plt.ylabel('prodvalue')
plt.title('totalprod-prodvalue Scatter Plot')
df.isnull().sum()  #null value control
                   #There is no missing value in the dataset 
sns.boxplot(x=df['priceperlb'])                   #Value Value Detection
                                                  #data gathered at a certain point
                                                  #the data set is significantly negative
#Create New Attribute
stockFilt = (df['stocks'])
df['stockFilt']=stockFilt/1000 
df
#Normalization
from sklearn import preprocessing
x = df[['totalprod']].values.astype(float)       #Normalize the 'totalprod' attribute

min_max_scaler = preprocessing.MinMaxScaler()    #We use the MinMax normalization method for normalization
x_scaled = min_max_scaler.fit_transform(x)
df['totalprod-2'] =pd.DataFrame(x_scaled)        #New 'totalprod-2' column

df.head(12)
#Creating df2
df2=df[['numcol','yieldpercol','totalprod','stocks','priceperlb','prodvalue']]
df2.head()
#Creating df3
df3=df['year']
df3.head()
X = df2.iloc[::].values                   #Select relevant attribute values for training
Y = df3.iloc[::].values                   #Select classification attribute values
Y
#½20 test and ½80 train(Multiple Lineer Regression)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
#model import operation
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
# Predicting the Test set results
y_pred = model.predict(X_test)
#Results Mulitple Lineer Regression
from sklearn import metrics
from sklearn.metrics import accuracy_score
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#½30 test and ½70 train(Multiple Lineer Regression)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#Results Mulitple Lineer Regression
from sklearn import metrics
from sklearn.metrics import accuracy_score
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#Results Multiple Lineer Regression
#Mean Squared Error(MSE) is quite similar to Mean Absolute Error, the only difference being that MSE takes the average of the square of the difference between the original values and the predicted values
#Mean Absolute Error is the average of the difference between the Original Values and the Predicted Values. It gives us the measure of how far the predictions were from the actual output
#Confusion Matrix as the name suggests gives us a matrix as output and describes the complete performance of the model.
#Naive Bayes 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import model_selection

X2 = df2.iloc[::].values                 #Select relevant attribute values for training
Y2 = df3.iloc[::].values                 #Select classification attribute values

X2
#test and train
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X2, Y2, test_size=validation_size, random_state=seed)
#Creation of Naive Bayes model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

cv_results
msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
msg
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, Y2, test_size = 0.2, random_state = 0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
from sklearn import metrics
from sklearn.metrics import accuracy_score
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("ACC: ",accuracy_score(y_pred,y_test))
#NAIVE BAYES RESULTS
#precision=The number of correct positive results is divided by the number of positive results estimated by the classifier.
#low value of recall value
#F1 Score tries to find the balance between precision and recall.The F1 score is low in Naive bayes
#Recall :It is the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive).
#Recall rate efficiency is high only in 2001
#Confusion Matrix as the name suggests gives us a matrix as output and describes the complete performance of the model.
#INTERPRATATION OF RESULTS
#We used 2 different model training in this data set.(Naive Bayes,Multiple Lineer Regression)
#ACC value in model training is not high
#Confusion Matrix values vary by years.
#Other outcome measures differ slightly in test and train operations.
