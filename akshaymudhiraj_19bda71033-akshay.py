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
import os
import gc
import matplotlib.pyplot as plt
import dask.dataframe as dd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score,roc_curve,recall_score,classification_report,mean_squared_error,confusion_matrix
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
# import lightgbm as lgb3
from contextlib import contextmanager
import time
import threading
import random
import seaborn as sns
from sklearn.model_selection import train_test_split as model_tts
test = pd.read_csv("/kaggle/input/pack123/Test.csv")
test
train = pd.read_csv("/kaggle/input/packaging/Train.csv")
train
train.info()
train.describe()
Final=train
feature_col=['currentBack','motorTempBack','positionBack','trackingDeviationBack','currentFront','motorTempFront','positionFront','trackingDeviationFront']
x=Final[feature_col]
y=Final.flag
X_train,X_test,y_train,y_test=model_tts(x,y,test_size=0.45,random_state=2)
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
import statsmodels.api as sm
X2 = sm.add_constant(X_test)
est = sm.OLS(y_test, X2)
est2 = est.fit()
print(est2.summary())
from sklearn import metrics
cnf_metric= metrics.confusion_matrix(y_test,y_pred)
cnf_metric
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#names of classes
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_metric), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
from statsmodels.api import OLS
from sklearn import metrics
cnf_metric_log= metrics.confusion_matrix(y_test,y_pred)
cnf_metric_log
print(cnf_metric_log)
total1=sum(sum(cnf_metric_log))
#####from confusion matrix calculate accuracy
accuracy1=(cnf_metric[0,0]+cnf_metric[1,1])/float(total1)
print ('Accuracy : ', accuracy1)
sensitivity1 = cnf_metric[0,0]/float((cnf_metric[0,0]+cnf_metric[0,1]))
print('Sensitivity : ', sensitivity1 )
specificity1 = cnf_metric[1,1]/float((cnf_metric[1,0]+cnf_metric[1,1]))
print('Specificity : ', specificity1)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=300)
#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
y_pred1=clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred1))
a=metrics.accuracy_score(y_test, y_pred1)
cnf_metric= metrics.confusion_matrix(y_test,y_pred1)
cnf_metric
print(cnf_metric)
total1=sum(sum(cnf_metric))
#####from confusion matrix calculate accuracy
accuracy1=(cnf_metric[0,0]+cnf_metric[1,1])/float(total1)
print ('Accuracy : ', accuracy1)
sensitivity1 = cnf_metric[0,0]/float((cnf_metric[0,0]+cnf_metric[0,1]))
print('Sensitivity : ', sensitivity1 )
specificity1 = cnf_metric[1,1]/float((cnf_metric[1,0]+cnf_metric[1,1]))
print('Specificity : ', specificity1)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_metric), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
a=clf.predict_proba(X_test)[:,1]
a
y_pred
fpr, tpr, thresholds =roc_curve(y_test, y_pred1,drop_intermediate=False)
plt.figure()

##Adding the ROC
plt.plot(fpr, tpr, color='red',
lw=2, label='ROC curve')
##Random FPR and TPR
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
##Title and label
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()
features=['currentBack','motorTempBack','positionBack','trackingDeviationBack','currentFront','motorTempFront','positionFront','trackingDeviationFront']

test['flag']=clf.predict(test[feature_col])

test.head()
test['Sl.No'] = test['timeindex']

test.head()
submission = test[['Sl.No','flag']]

submission.to_csv("submission.csv", index=False)

submission.head(20)
