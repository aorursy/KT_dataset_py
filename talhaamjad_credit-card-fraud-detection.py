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
# import the necessary packages 

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 

from matplotlib import gridspec 
# Load the dataset from the csv file using pandas 

# best way is to mount the drive on colab and  

# copy the path for the csv file 

data = pd.read_csv("../input/creditcardfraud/creditcard.csv")

data
# Grab a peek at the data 

data.head() 
# Print the shape of the data 

# data = data.sample(frac = 0.1, random_state = 48) 

print(data.shape) 

print(data.describe()) 
# Determine number of fraud cases in dataset 

fraud = data[data['Class'] == 1] 

valid = data[data['Class'] == 0] 

outlierFraction = len(fraud)/float(len(valid)) 

print(outlierFraction) 

print('Fraud Cases: {}'.format(len(data[data['Class'] == 1]))) 

print('Valid Transactions: {}'.format(len(data[data['Class'] == 0]))) 
print('Amount details of the fraudulent transaction') 

fraud.Amount.describe() 
print('details of valid transaction') 

valid.Amount.describe() 
# Correlation matrix 

corrmat = data.corr() 

fig = plt.figure(figsize = (12, 9)) 

sns.heatmap(corrmat, vmax = .8, square = True) 

plt.show() 
# dividing the X and the Y from the dataset 

X = data.drop(['Class'], axis = 1) 

Y = data["Class"] 

print(X.shape) 

print(Y.shape) 

# getting just the values for the sake of processing  

# (its a numpy array with no columns) 

xData = X.values 

yData = Y.values 
# Using Skicit-learn to split data into training and testing sets 

from sklearn.model_selection import train_test_split 

# Split the data into training and testing sets 

xTrain, xTest, yTrain, yTest = train_test_split( 

		xData, yData, test_size = 0.2, random_state = 42) 

# Building the Random Forest Classifier (RANDOM FOREST) 

from sklearn.ensemble import RandomForestClassifier 

# random forest model creation 

rfc = RandomForestClassifier() 

rfc.fit(xTrain, yTrain) 

# predictions 

yPred = rfc.predict(xTest) 

# Evaluating the classifier 

# printing every score of the classifier 

# scoring in anything 

from sklearn.metrics import classification_report, accuracy_score  

from sklearn.metrics import precision_score, recall_score 

from sklearn.metrics import f1_score, matthews_corrcoef 

from sklearn.metrics import confusion_matrix 

  

n_outliers = len(fraud) 

n_errors = (yPred != yTest).sum() 

print("The model used is Random Forest classifier") 

  

acc = accuracy_score(yTest, yPred) 

print("The accuracy is {}".format(acc)) 

  

prec = precision_score(yTest, yPred) 

print("The precision is {}".format(prec)) 

  

rec = recall_score(yTest, yPred) 

print("The recall is {}".format(rec)) 

  

f1 = f1_score(yTest, yPred) 

print("The F1-Score is {}".format(f1)) 

  

MCC = matthews_corrcoef(yTest, yPred) 

print("The Matthews correlation coefficient is{}".format(MCC)) 

# printing the confusion matrix 

LABELS = ['Normal', 'Fraud'] 

conf_matrix = confusion_matrix(yTest, yPred) 

plt.figure(figsize =(12, 12)) 

sns.heatmap(conf_matrix, xticklabels = LABELS, 

			yticklabels = LABELS, annot = True, fmt ="d"); 

plt.title("Confusion matrix") 

plt.ylabel('True class') 

plt.xlabel('Predicted class') 

plt.show() 
