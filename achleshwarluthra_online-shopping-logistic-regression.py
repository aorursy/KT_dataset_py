import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn import metrics
from sklearn.metrics import confusion_matrix
%matplotlib inline
#read the data
df = pd.read_csv('/kaggle/input/conversion_data.csv')
#sneak peek into the data
df.head()
#more details about df
df.info()
#target class frequency
df.converted.value_counts()
10200/(306000+10200) #Only 3% converted
df.shape
#dummy data for categorical variables
df = pd.get_dummies(df, columns=['country','source'])
df.head()
from sklearn import preprocessing
age = df['age'].values.astype('float')
age.reshape(1,-1)
age = pd.DataFrame(age , columns = ['age'])
age.head()
age = preprocessing.normalize(age)
age = pd.DataFrame(age , columns = ['age'])
age.head()
df['age'] = age['age']
df.head()
X = df.drop(columns = ['converted'])
y = df.converted
print (X.shape, y.shape)
#import model specific libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#Split the data into training and test data (70/30 ratio)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=100, stratify=y)
#validate the shape of train and test dataset
print (X_train.shape)
print (y_train.shape)

print (X_test.shape)
print (y_test.shape)
y_train.value_counts()
y_test.value_counts()
#fit the logisitc regression model on training dataset 
logreg = LogisticRegression(solver = 'lbfgs', max_iter=300)
logreg.fit(X_train,y_train)
y_train_pred = logreg.predict(X_train)
y_test_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_train, y_train_pred))
print(metrics.accuracy_score(y_test, y_test_pred))
train_auc = metrics.roc_auc_score(y_train, y_train_pred)
test_auc = metrics.roc_auc_score(y_test, y_test_pred)
train_confusion = confusion_matrix(y_train, y_train_pred)
train_TP = train_confusion[1, 1]
train_TN = train_confusion[0, 0]
train_FP = train_confusion[0, 1]
train_FN = train_confusion[1, 0]
test_confusion = confusion_matrix(y_test, y_test_pred)
test_TP = test_confusion[1, 1]
test_TN = test_confusion[0, 0]
test_FP = test_confusion[0, 1]
test_FN = test_confusion[1, 0]
print(train_confusion)
print(test_confusion)
train_fpr, train_tpr, train_thresholds = metrics.roc_curve(y_train, y_train_pred)
test_fpr, test_tpr, test_thresholds = metrics.roc_curve(y_test, y_test_pred)
plt.title('Receiver Operating Characteristic')
plt.plot(train_fpr, train_tpr, 'b',label = 'auc: ' + str(train_auc))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.title('Receiver Operating Characteristic')
plt.plot(test_fpr, test_tpr, 'b',label = 'auc: ' + str(test_auc))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend(loc = 'lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()