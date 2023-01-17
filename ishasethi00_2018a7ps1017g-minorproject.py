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
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('/kaggle/input/minor-project-2020/train.csv')
df.head()
df.shape
# Correlation matrix
corr = df.corr()
corr
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize=(50, 50))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap='RdBu', mask=mask, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.show()
# Drop 'id' and 'target'
df1=df.drop(['id','target'],axis=1)
df1.head()
df1.shape
X=df1
y=df['target']
pd.value_counts(y).plot.bar()
plt.title('target histogram')
plt.xlabel('Target')
plt.ylabel('Frequency')
df['target'].value_counts()
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
from sklearn.metrics import auc, roc_curve

clf = LogisticRegression(max_iter=1000).fit(X_train,y_train)
y_predict_proba=clf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_predict_proba)
roc_auc = auc(fpr,tpr)
roc_auc
# Using GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import auc, roc_auc_score, roc_curve

lr = LogisticRegression(max_iter=1000)
grid_values={'C':[8,9,10,11,12]}

grid_clf = GridSearchCV(lr,param_grid=grid_values,scoring='roc_auc')
grid_clf.fit(X_train,y_train)
y_grid=grid_clf.predict_proba(X_test)[:,1]

print('Test set AUC: ',roc_auc_score(y_test,y_grid))
print('Grid best parameter (max AUC): ',grid_clf.best_params_)
print('Grid best score (AUC): ',grid_clf.best_score_)
# Applying value of C
clf1 = LogisticRegression(max_iter=1000,C=9).fit(X_train,y_train)
y_predict_proba1=clf1.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_predict_proba1)
roc_auc1 = auc(fpr,tpr)
roc_auc1
# Plotting ROC curve
plt.title('ROC Curve - Minor Project')
plt.plot(fpr, tpr, 'b', label='AUC = {:0.5f}'.format(roc_auc1))
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
df2 = pd.read_csv('/kaggle/input/minor-project-2020/test.csv')
df2.head()
df2.shape
New_X=df2.drop('id',axis=1)
New_X.shape
target=clf1.predict_proba(New_X)[:,1]
t=pd.DataFrame(target)
df2['target']=t
df3=df2[['id','target']]
df3.head()
df3.to_csv('./submisson2.csv',index=False) 








