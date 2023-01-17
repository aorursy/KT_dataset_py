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
# Let's import some libraries

import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset:

df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df.head()
# shape of the dataset
print("Shape of the Dataset:",df.shape)
# describe
df.describe()
# info
df.info()
# columns
df.columns
# Let's plot the y variable

plt.figure(figsize=(5,4))
sns.distplot(df['Outcome'],color = 'red')
plt.show()
# Let's plot the relation X variables and y variable

plt.figure(figsize=(5,4))
sns.barplot(df['Outcome'],df['Outcome'].value_counts(),color = 'red')
plt.show()
df['Outcome'].value_counts()
# Let's start modelling
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report 
from sklearn.preprocessing import StandardScaler
## Let'sdivide the dataset into X and y
y = df.pop('Outcome')
X = df
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 20)
print("X Train shape :", X_train.shape)
print("X Test shape  :", X_test.shape)
print("Y Train shape :", y_train.shape)
print("Y Test shape  :", y_test.shape)
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
result = pd.DataFrame({'Actual ': y_test,'Predicted':y_pred})
result.head()
cm = confusion_matrix(y_pred,y_test)
cr = classification_report(y_pred,y_test)
acc = accuracy_score(y_pred,y_test)

print("**** Before Scaling ****")
print("Accuracy After Scaling:", acc)
print("Classification Report :\n",cr)
print("confusion Matrix      :\n",cm)
tn = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tp = cm[1][1]

recall = tp/(tp+fn)*100
precision = tp/(tp+fp)*100
specificity = tn/(tn+fp)*100

print("**** Before Scaling ****")
print("Recall/Sensitivity :",recall)
print("Precision          :",precision)
print("Specificity        :",specificity)

# Scaling the X variables
scaler = StandardScaler()
X_scale = scaler.fit_transform(X)
X_train_scaled,X_test_scaled,y_train_scaled,y_test_scaled = train_test_split(X_scale,y,test_size = 0.3,random_state = 20)
print("X Train shape :", X_train_scaled.shape)
print("X Test shape  :", X_test_scaled.shape)
print("Y Train shape :", y_train_scaled.shape)
print("Y Test shape  :", y_test_scaled.shape)
logreg_scale = LogisticRegression()
logreg_scale.fit(X_train_scaled,y_train_scaled)
y_pred_scale = logreg_scale.predict(X_test_scaled)
result = pd.DataFrame({'Actual ': y_test_scaled,'Predicted':y_pred_scale})
result.head()
cm_scale = confusion_matrix(y_pred_scale,y_test_scaled)
cr_scale = classification_report(y_pred_scale,y_test_scaled)
acc_scale = accuracy_score(y_pred_scale,y_test_scaled)

print("**** After Scaling ****")
print("Accuracy              :", acc_scale)
print("Classification Report :\n",cr_scale)
print("confusion Matrix      :\n",cm_scale)
tn_scale = cm_scale[0][0]
fp_scale = cm_scale[0][1]
fn_scale = cm_scale[1][0]
tp_scale = cm_scale[1][1]

recall_scale = tp_scale/(tp_scale  + fn_scale)*100
precision_scale = tp_scale/(tp_scale + fp_scale)*100
specificity_scale = tn_scale/(tn_scale + fp_scale)*(100)

print("**** After Scaling ****")
print("Recall/Sensitivity :",recall_scale)
print("Precision          :",precision_scale)
print("Specificity        :",specificity_scale)

from imblearn.over_sampling import SMOTE
# transform the dataset
oversample = SMOTE(random_state=42)
X_smote, y_smote = oversample.fit_resample(X, y)
X_train_smote,X_test_smote,y_train_smote,y_test_smote = train_test_split(X_smote,y_smote,test_size = 0.3,random_state = 20)
print("X Train shape :", X_train_smote.shape)
print("X Test shape  :", X_test_smote.shape)
print("Y Train shape :", y_train_smote.shape)
print("Y Test shape  :", y_test_smote.shape)
logreg_smote = LogisticRegression()
logreg_smote.fit(X_train_smote,y_train_smote)
y_pred_smote = logreg_smote.predict(X_test_smote)
result = pd.DataFrame({'Actual ': y_test_smote,'Predicted':y_pred_smote})
result.head()
cm_smote = confusion_matrix(y_pred_smote,y_test_smote)
cr_smote = classification_report(y_pred_smote,y_test_smote)
acc_smote = accuracy_score(y_pred_smote,y_test_smote)

print("**** After applying SMOTE ****")
print("Accuracy              :", acc_smote)
print("Classification Report :\n",cr_smote)
print("confusion Matrix      :\n",cm_smote)
tn_smote = cm_smote[0][0]
fp_smote = cm_smote[0][1]
fn_smote = cm_smote[1][0]
tp_smote = cm_smote[1][1]

recall_smote = tp_smote/(tp_smote + fn_smote)*100
precision_smote = tp_smote/(tp_smote + fp_smote)*100
specificity_smote = tn_smote/(tn_smote + fp_smote)*(100)

print("**** After applying SMOTE ****")
print("Recall/Sensitivity :",recall_smote)
print("Precision          :",precision_smote)
print("Specificity        :",specificity_smote)

