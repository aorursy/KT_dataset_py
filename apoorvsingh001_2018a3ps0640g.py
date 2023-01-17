
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
import seaborn as sns
import sklearn
from sklearn import svm, preprocessing
import math
import matplotlib.pyplot as plt

df = pd.read_csv("/kaggle/input/minor-project-2020/train.csv")

df

df.drop('id',axis=1,inplace=True)



from sklearn.preprocessing import StandardScaler

X= df.drop(['target'], axis=1).values
y=df['target']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=121)

print(len(X_train), len(X_test))



from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

scaled_X_train = preprocessing.normalize(X_train)
scaled_X_test= preprocessing.normalize(X_test)
#scaler = preprocessing.MinMaxScaler() 
#scalar = StandardScaler()
#scaled_X_train = scaler.fit_transform(X_train)
#scaled_X_test = scaler.fit_transform(X_test)

             #Instantiate the scaler
#scaled_X_train = scaler.fit_transform(X_train)     #Fit and transform the data

#scaled_X_train


from sklearn import linear_model, feature_selection
logr = linear_model.LogisticRegression(max_iter=1000)
logr.fit(scaled_X_train, y_train)


logr.score(scaled_X_test, y_test)
y_pred = logr.predict_proba(scaled_X_test)
logr.score(scaled_X_test, y_test)


prediction =[]
for i in range (160000):
    prediction.append(y_pred[i][1])
prediction=np.array(prediction)
prediction.resize([160000,1])

from sklearn.metrics import roc_curve, auc
FPR, TPR, _ = roc_curve(y_test, prediction)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)

from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
plot_confusion_matrix(logr, scaled_X_test, y_test, cmap = plt.cm.Blues)
dft = pd.read_csv("kaggle/input/minor-project-2020/test.csv")
Xnew= dft.drop(['id'], axis=1).values
scaled_Xnew= preprocessing.normalize(Xnew)
#scaled_Xnew = scaler.fit_transform(X)
y_pred_new = logr.predict_proba(scaled_Xnew)

predictionn =[]
for i in range (200000):
    predictionn.append(y_pred_new[i][1])
predictionn=np.array(predictionn)
predictionn.resize([200000])
dft['target']=predictionn



result = pd.concat([dft['id'], dft['target']], axis=1, sort=False)
#result.to_csv('/kaggle/input/minor-project-2020/sample_submission.csv', index =False)
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred1 = dt.predict(X_test)
print("Accuracy is : {}".format(dt.score(scaled_X_test, y_test)))
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
print("Confusion Matrix: ")

print(confusion_matrix(y_test, y_pred1))

from sklearn.metrics import roc_curve, auc
plt.style.use('seaborn-pastel')

FPR, TPR, _ = roc_curve(y_test, y_pred1)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)
from sklearn.model_selection import GridSearchCV

parameters = {'criterion': ("gini", "entropy"), 'max_depth': (50,300)}

dt_cv = DecisionTreeClassifier()

clf = GridSearchCV(dt_cv, parameters, verbose=1)

clf.fit(scaled_X_train, y_train)

clf.score(scaled_X_test, y_test)
y_pred2 = clf.predict(X_test)

FPR, TPR, _ = roc_curve(y_test, y_pred2)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)

