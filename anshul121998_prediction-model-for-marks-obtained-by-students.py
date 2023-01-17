import pandas as pd
import math
import numpy as np
df = pd.read_csv("../input/StudentsPerformance.csv")
df.head()
df['gender'] = df['gender'].map({'female': 0, 'male': 1})
df['race/ethnicity'] = df['race/ethnicity'].map({'group E' : 4, 'group D' : 3, 'group C' : 2, 'group B' : 1, 'group A' : 0})
df['test preparation course'] = df['test preparation course'].map({'completed' : 1, 'none' : 0})
df['lunch'] = df['lunch'].map({'standard' : 1, 'free/reduced' : 0})
df['parental level of education'] = df['parental level of education'].map({"master's degree" : 5, "bachelor's degree" : 4, "associate's degree" : 3, 'high school' : 2, 'some high school' : 1, 'some college' : 0})
df.head()
for i in range (0,len(df)):
    if(df['math score'][i]<10):
        df['math score'][i] = 0
    elif(df['math score'][i]<20):
        df['math score'][i] = 1
    elif(df['math score'][i]<30):
        df['math score'][i] = 2
    elif(df['math score'][i]<40):
        df['math score'][i] = 3
    elif(df['math score'][i]<50):
        df['math score'][i] = 4
    elif(df['math score'][i]<60):
        df['math score'][i] = 5
    elif(df['math score'][i]<70):
        df['math score'][i] = 6
    elif(df['math score'][i]<80):
        df['math score'][i] = 7
    elif(df['math score'][i]<90):
        df['math score'][i] = 8
    elif(df['math score'][i]<=100):
        df['math score'][i] = 9
    else:
        pass
    
    if(df['reading score'][i]<10):
        df['reading score'][i] = 0
    elif(df['reading score'][i]<20):
        df['reading score'][i] = 1
    elif(df['reading score'][i]<30):
        df['reading score'][i] = 2
    elif(df['reading score'][i]<40):
        df['reading score'][i] = 3
    elif(df['reading score'][i]<50):
        df['reading score'][i] = 4
    elif(df['reading score'][i]<60):
        df['reading score'][i] = 5
    elif(df['reading score'][i]<70):
        df['reading score'][i] = 6
    elif(df['reading score'][i]<80):
        df['reading score'][i] = 7
    elif(df['reading score'][i]<90):
        df['reading score'][i] = 8
    elif(df['reading score'][i]<=100):
        df['reading score'][i] = 9
    else:
        pass
    
    if(df['writing score'][i]<10):
        df['writing score'][i] = 0
    elif(df['writing score'][i]<20):
        df['writing score'][i] = 1
    elif(df['writing score'][i]<30):
        df['writing score'][i] = 2
    elif(df['writing score'][i]<40):
        df['writing score'][i] = 3
    elif(df['writing score'][i]<50):
        df['writing score'][i] = 4
    elif(df['writing score'][i]<60):
        df['writing score'][i] = 5
    elif(df['writing score'][i]<70):
        df['writing score'][i] = 6
    elif(df['writing score'][i]<80):
        df['writing score'][i] = 7
    elif(df['writing score'][i]<90):
        df['writing score'][i] = 8
    elif(df['writing score'][i]<=100):
        df['writing score'][i] = 9
    else:
        pass
df.head()
df.isnull().values.any()
from sklearn.utils import shuffle
df= shuffle(df).reset_index(drop=True)
X = df.drop(['math score','reading score','writing score'],1)
X.head()
y_math = df["math score"]
y_reading = df["reading score"]
y_writing = df["writing score"]
print(X.shape)
print(y_math.shape)
print(y_reading.shape)
print(y_writing.shape)
from sklearn.model_selection import train_test_split 
X_train_math, X_test_math, y_train_math, y_test_math = train_test_split(X, y_math, test_size=0.15, random_state=100)
X_train_reading, X_test_reading, y_train_reading, y_test_reading = train_test_split(X, y_reading, test_size=0.15, random_state=100)
X_train_writing, X_test_writing, y_train_writing, y_test_writing = train_test_split(X, y_writing, test_size=0.15, random_state=100)
from sklearn import tree
LogisticRegressionModel_math = tree.DecisionTreeClassifier()
LogisticRegressionModel_math.fit(X_train_math, y_train_math)

LogisticRegressionModel_reading = tree.DecisionTreeClassifier()
LogisticRegressionModel_reading.fit(X_train_reading, y_train_reading)

LogisticRegressionModel_writing = tree.DecisionTreeClassifier()
LogisticRegressionModel_writing.fit(X_train_writing, y_train_writing)
training_accuracy_math = LogisticRegressionModel_math.score(X_train_math, y_train_math)
print ('Training Accuracy for Math :',training_accuracy_math)

training_accuracy_reading = LogisticRegressionModel_reading.score(X_train_reading, y_train_reading)
print ('Training Accuracy for Reading :',training_accuracy_reading)

training_accuracy_writing = LogisticRegressionModel_writing.score(X_train_writing, y_train_writing)
print ('Training Accuracy for Writing :',training_accuracy_writing)
test_accuracy_math = LogisticRegressionModel_math.score(X_test_math,y_test_math)
print('Accuracy of the model on unseen test data for math score: ',test_accuracy_math)

test_accuracy_reading = LogisticRegressionModel_reading.score(X_test_reading,y_test_reading)
print('Accuracy of the model on unseen test data for reading score: ',test_accuracy_reading)

test_accuracy_writing = LogisticRegressionModel_writing.score(X_test_writing,y_test_writing)
print('Accuracy of the model on unseen test data for writing score: ',test_accuracy_writing)

import graphviz 
dot_data = tree.export_graphviz(LogisticRegressionModel_math, out_file=None) 
graph = graphviz.Source(dot_data) 
graph

