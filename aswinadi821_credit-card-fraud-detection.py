import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import gridspec


df = pd.read_csv("../input/creditcard/creditcard.csv")

df.head()
print(df.shape)

print(df.describe())
fraud = df[df['Class']==1]

valid = df[df['Class']==0]

outlierfraction=len(fraud)/float(len(valid))

print(outlierfraction)

print("Fraud cases: {}".format(len(df[df['Class']==1])))

print('Valid Transacions: {}'.format(len(df[df['Class']==0])))
print('Amount details of the fraudulent transaction') 

fraud.Amount.describe() 
print('details of valid transaction') 

valid.Amount.describe() 
corrmat=df.corr()

fig=plt.figure(figsize=(13,8))

sns.heatmap(corrmat,vmax = .8, square =True)

plt.show()
X=df.drop(['Class'], axis=1)

Y=df['Class']

print(X.shape)

print(Y.shape)

xData = X.values

yData = Y.values
from sklearn.model_selection import train_test_split

xTrain,xTest, yTrain,yTest=train_test_split(xData,yData,test_size=0.2,random_state=42)
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(xTrain,yTrain)

yPred=rfc.predict(xTest)

yPred
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
LABELS = ['Normal', 'Fraud'] 

conf_matrix = confusion_matrix(yTest, yPred) 

plt.figure(figsize =(10, 10)) 

sns.heatmap(conf_matrix, xticklabels = LABELS,  

            yticklabels = LABELS, annot = True, fmt ="d"); 

plt.title("Confusion matrix") 

plt.ylabel('True class') 

plt.xlabel('Predicted class') 

plt.show() 