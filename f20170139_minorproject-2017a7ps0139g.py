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
import matplotlib.pyplot as plt
import seaborn as sns 

%matplotlib inline
df_train = pd.read_csv("/kaggle/input/minor-project-2020/train.csv")
#Checking skewness of data
df_train.skew(axis = 0, skipna = True) 
y=df_train[['target']]
X=df_train.drop(['id','target'],axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=121)
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scaled_X_train = scalar.fit_transform(X_train)
scaled_X_test = scalar.transform(X_test)
# Feature selection using RFE


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,roc_curve, roc_auc_score, recall_score
from sklearn.feature_selection import RFE
binaryclass_model_LR = LogisticRegression(random_state=13, penalty = 'l2', fit_intercept=True, solver= 'lbfgs', C=1, class_weight={0:0.0018,1:0.998}, max_iter=10000 )

rfe = RFE(estimator=binaryclass_model_LR)

rfe.fit(scaled_X_train,y_train)
#predicting probabilities
y_pred=rfe.predict_proba(scaled_X_test)
probs = y_pred[:, 1]
# Evaluating It


print(f'Area Under Curve: {roc_auc_score(y_test, probs)}')

# ROC curve

from sklearn.metrics import roc_curve, auc
plt.style.use('seaborn-pastel')

FPR, TPR, _ = roc_curve(y_test, probs)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)

plt.figure(figsize =[11,9])
plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('ROC Curve', fontsize= 18)
plt.show()
df_test = pd.read_csv("/kaggle/input/minor-project-2020/test.csv")
X_t=df_test.drop(['id'],axis=1)
scalar = StandardScaler()
scaled_X_train = scalar.fit_transform(X)
scaled_X_test = scalar.transform(X_t)
#Grid search on solver,C,class weights

binaryclass_model_LR = LogisticRegression(random_state=13, penalty = 'l2', fit_intercept=True, solver= 'lbfgs', C=1, class_weight={0:0.0018,1:0.998}, max_iter=10000 )

rfe = RFE(estimator=binaryclass_model_LR)

rfe.fit(scaled_X_train,y)
y_pred=rfe.predict_proba(scaled_X_test)
probs = y_pred[:, 1]
#Writing output to dataframe

pd.DataFrame({'id':df_test['id'],'target':probs}).to_csv('/kaggle/working/submit.csv',index=False)
from IPython.display import FileLink
FileLink(r'submit.csv')