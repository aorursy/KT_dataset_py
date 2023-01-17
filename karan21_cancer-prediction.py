import numpy as np #Linear Algebra
import pandas as pd #Data Manipulation

import matplotlib.pyplot as plt #Plotting Data
from sklearn.model_selection import train_test_split #Split Data into test and train 
from sklearn.linear_model import LogisticRegression #
from sklearn import metrics
df = pd.read_csv("../input/data.csv",header = 0)
df.head()
df.drop('id', axis = 1, inplace = True)
df.drop('Unnamed: 32', axis = 1, inplace = True)

df.diagnosis.unique()
df.head()
# split dataframe into two based on diagnosis
dfM=df[df['diagnosis'] =='M']
dfB=df[df['diagnosis'] =='B']
plt.show(plt.hist(df['radius_mean'], bins = 25 ), plt.hist(dfB['radius_mean'], bins = 25 ))
plt.show(plt.hist(df['radius_mean'], bins = 25 ), plt.hist(dfM['radius_mean'], bins = 25 ))


traindf, testdf = train_test_split(df, test_size = 0.3)
traindf.shape
testdf.shape
#Generic function for making a classification model and accessing the performance. 

def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print("Accuracy : %s" % "{0:.3%}".format(accuracy))
  print(predictions.size)
  
predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']
outcome_var='diagnosis' 
model=LogisticRegression()
classification_model(model,traindf,predictor_var,outcome_var)