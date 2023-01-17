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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt,matplotlib.image as mping
import seaborn as sns
df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df_test =  pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
df_train.info()
df_test.info()
df_train.iloc[:,0].unique()
sns.countplot(df_train.iloc[:,0])
y = df_train.iloc[:,0]
X = df_train.iloc[:,1:]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 , random_state = 42)
y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier().fit(X_train,y_train)
y_pred = tree.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report,plot_confusion_matrix,confusion_matrix
accuracy_score(y_test,y_pred)
from sklearn.ensemble import RandomForestClassifier
random_classifier =  RandomForestClassifier(random_state=42).fit(X_train,y_train)
random_classifier.score(X_test,y_test)
from sklearn.ensemble import BaggingClassifier
bagging_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100,oob_score=True,random_state=42).fit(X_train,y_train)
bagging_clf.score(X_test,y_test)
from sklearn.svm import SVC
svm_parameter = SVC(C= 10,degree = 3,gamma = 'scale',random_state=42).fit(X_train,y_train)
y_pred = svm_parameter.predict(X_test)
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True)
print(classification_report(y_test,y_pred))
## here we can see that the svm has get most accuracy
## we are going to use svm model for predication
df_test
X_test_data = df_test.iloc[:,:]
y_pred_data = svm_parameter.predict(X_test_data)
y_pred_data = y_pred_data.reshape(-1,1)
df_sample_data = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
df_pred = pd.DataFrame(y_pred_data,columns=['Label'])
df_sample_data = df_sample_data.drop(columns=['Label'])
sample_data = pd.concat([df_sample_data,df_pred],axis=1)
sample_data.info()
sample_data.to_csv('result.csv')