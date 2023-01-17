import itertools
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib import gridspec
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
#%matplotlib inline
df=pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.head(5)
X=df.iloc[:,:-1].values
y=df[['Class']].values
y[0:10]
#Corelation Matrix
corrmat=df.corr()
fig = plt.figure(figsize = (12, 9)) 
sns.heatmap(corrmat, vmax = .8, square = True) 
plt.show()
#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)
from sklearn.tree import DecisionTreeClassifier
FDTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
FDTree # it shows the default parameters
FDTree.fit(X_train,y_train)
#Prediction
predTree = FDTree.predict(X_test)
print (predTree [0:5])
print (y_test [0:5])
#Evaluation
# Evaluating the classifier 
# printing every score of the classifier 
# scoring in anything 
from sklearn.metrics import classification_report, accuracy_score  
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import f1_score, matthews_corrcoef 
from sklearn.metrics import confusion_matrix 
  
print("The model used is Decision Tree classifier") 
  
acc = accuracy_score(y_test, predTree) 
print("The accuracy is {}".format(acc)) 
  
prec = precision_score(y_test, predTree) 
print("The precision is {}".format(prec)) 
  
rec = recall_score(y_test, predTree) 
print("The recall is {}".format(rec)) 
  
f1 = f1_score(y_test, predTree) 
print("The F1-Score is {}".format(f1)) 
  
MCC = matthews_corrcoef(y_test, predTree) 
print("The Matthews correlation coefficient is{}".format(MCC)) 
# printing the confusion matrix 
LABELS = ['Normal', 'Fraud'] 
conf_matrix = confusion_matrix(y_test, predTree) 
plt.figure(figsize =(12, 12)) 
sns.heatmap(conf_matrix, xticklabels = LABELS,  
            yticklabels = LABELS, annot = True, fmt ="d"); 
plt.title("Confusion matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show() 