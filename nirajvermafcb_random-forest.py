# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
# Reading the given csv file
data = pd.read_csv('../input/creditcard.csv')
print(data.head())
# Checking whether the dataset has any null values
data.isnull().sum()
data.shape
data.describe()
data['Time']
# Dropping the **Time** and **Amount** column
data.drop(['Time'], axis=1, inplace=True)
data.drop(['Amount'], axis=1, inplace=True)
data.head()
print(data.groupby('Class').size())
data.dtypes
from sklearn.utils import shuffle
data=shuffle(data)
data.head()
data1= data.iloc[:,0:4]
data1.head()
plt.figure(figsize=(20,10))
data1.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
data2= data.iloc[:,4:8]
data2.head()
plt.figure(figsize=(20,10))
data2.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
data3= data.iloc[:,8:12]
data3.head()
plt.figure(figsize=(20,10))
data3.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
data4= data.iloc[:,12:16]
data4.head()
plt.figure(figsize=(20,10))
data4.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
data5= data.iloc[:,16:20]
data5.head()
plt.figure(figsize=(20,10))
data5.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
data6= data.iloc[:,20:24]
data6.head()
plt.figure(figsize=(20,10))
data6.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
data7= data.iloc[:,24:28]
data7.head()
plt.figure(figsize=(20,10))
data7.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
X=data.iloc[:,1:28]
X.head()
y=data.iloc[:,-1]
y.head()
# Scale the data to be between -1 and 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X)
X
#Converting the strings into binary values
# Encode label category
# Fraud -> 1
# Nota fraud -> 0



from sklearn.preprocessing import LabelEncoder
fraud_encoder = LabelEncoder()
y = fraud_encoder.fit_transform(y)
y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)
# Initiating the model with default parameters. Then for tuning hyperprameters we are using GridsearchCV.
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

model_RR=RandomForestClassifier()

tuned_parameters = {'min_samples_leaf': range(5,10,5), 'n_estimators' : range(50,200,50),
                    'max_depth': range(5,15,5), 'max_features':range(5,20,5)
                    }
                   
model_RR.get_params().keys()
from sklearn.grid_search import RandomizedSearchCV

model= RandomizedSearchCV(model_RR, tuned_parameters,cv=10,scoring='accuracy',n_iter=5)
model.fit(X_train,y_train)
print(model_svm.grid_scores_)
print(model_svm.best_score_)
print(model_svm.best_params_)
y_pred = model.predict(X_test)
model.score(X_test,y_pred)
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix
sns.heatmap(confusion_matrix,linewidths=.5,annot=True,vmin=0.01,cmap='PuBuGn')
auc_roc=metrics.classification_report(y_test,y_pred)
auc_roc
auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc
TN=confusion_matrix[0][0]
FN=confusion_matrix[1][0]
TP=confusion_matrix[1][1]
FP=confusion_matrix[0][1]
TN,FN,TP,FP
TPR=(TP/(TP+FN))
TPR
FPR=(FP/(FP+TN))
FPR
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='black',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')