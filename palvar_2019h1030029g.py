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
import numpy as np
import pandas as pd
df = pd.read_csv('../input/minor-project-2020/train.csv')
df.head()
print(df.shape)
df.head
print(df.shape)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
#Y=[['target']]
#X=df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S']]
x=df.drop('target',axis=1)
y=df[['target']]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=121)
train_data = pd.concat([x_train, y_train], axis=1)
# separate minority and majority classes
negative = train_data[train_data.target==0]
positive = train_data[train_data.target==1]
neg_downsampled = resample(negative,
 replace=True, # sample with replacement
 n_samples=len(positive), # match number in minority class
 random_state=27) # reproducible results
# combine minority and downsampled majority
downsampled = pd.concat([positive, neg_downsampled])
# check new class counts
print(downsampled.target.value_counts())
print(downsampled.shape)
x_train=downsampled.drop('target',axis=1)
y_train=downsampled['target']
print(y_train[:4])
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
#Y=[['target']]
#X=df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S']]
x=df.drop('target',axis=1)
y=df[['target']]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=121)
train_data = pd.concat([x_train, y_train], axis=1)
# separate minority and majority classes
negative = train_data[train_data.target==0]
positive = train_data[train_data.target==1]
pos_upsampled = resample(positive,
 replace=True, # sample with replacement
 n_samples=len(negative), # match number in majority class
 random_state=27) # reproducible results
upsampled = pd.concat([negative, pos_upsampled])
# check new class counts
print(upsampled.target.value_counts())
print(upsampled.shape)
x_train=upsampled.drop('target',axis=1)
y_train=upsampled['target']
print(y_train[:4])
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
print(y_pred.shape)
df2=pd.read_csv('../input/minor-project-2020/test.csv')
data2=df2
print(data2.shape)
print(type(data2))
#print(data[:4,:])
x_test1= df2.values
#id1=data2[:,0]
y_pred1=dt.predict(x_test1)
print(y_pred1.shape)
    
    
submission=pd.DataFrame({
    "id": data2["id"],
    "target": y_pred1
})
submission.to_csv('sub_dt1.csv',index=False)
from sklearn.metrics import roc_curve, auc
plt.style.use('seaborn-pastel')

FPR, TPR, _ = roc_curve(y_pred, y_pred1)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)

plt.figure(figsize =[11,9])
plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('ROC for Minor project', fontsize= 18)
plt.show()
from sklearn.model_selection import GridSearchCV
parameters = {'criterion': ("gini", "entropy"), 'max_depth': (100,1)}

dt_cv = DecisionTreeClassifier()

clf = GridSearchCV(dt_cv, parameters, verbose=1)

clf.fit(x_train, y_train)
clf_y_pred = dt.predict(x_test)
print(clf_y_pred[:100])
clf_y_pred1=dt.predict(x_test1)
print(clf_y_pred1.shape)
    
submission=pd.DataFrame({
    "id": data2["id"],
    "target": clf_y_pred1[:]
})
submission.to_csv('sub5_grid.csv',index=False)
from sklearn.linear_model import LogisticRegression


logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)

lr_y_pred = logisticRegr.predict(x_test)
print(lr_y_pred[:100])

lr_y_pred1 = logisticRegr.predict(x_test1)
print(lr_y_pred1.shape)
print(lr_y_pred1[:4])
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(x_train,y_train)

rf_y_pred=clf.predict(x_test)
print(rf_y_pred)

rf_y_pred1=clf.predict(x_test1)
submission=pd.DataFrame({
    "id": data2["id"],
    "target": lr_y_pred1
})
submission.to_csv('sub_rf2.csv',index=False)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
gsc = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid={
            'C': [0.1, 1, 100, 1000],
            'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
from sklearn.metrics import roc_curve, auc
plt.style.use('seaborn-pastel')

FPR, TPR, _ = roc_curve(rf_y_pred, rf_y_pred1)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)

plt.figure(figsize =[11,9])
plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('ROC for Minor project', fontsize= 18)
plt.show()