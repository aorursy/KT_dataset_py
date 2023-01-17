# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os

print(os.listdir("../input"))
import pandas as pd

import numpy as np

from sklearn import preprocessing

import matplotlib.pyplot as plt 

plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import seaborn as sns

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)

import warnings

warnings.filterwarnings('ignore')



import pandas as pd



df = pd.read_csv("../input/HR-Employee-Attrition.csv")

df.head()
df.describe()
df.values
df.columns
df['BusinessTravel'].unique()
df.columns.values
df.isna().sum()
cat_col = df.select_dtypes(exclude=np.number).columns

num_col = df.select_dtypes(include=np.number).columns

print(cat_col)

print(num_col)
for i in cat_col:

    print(df[i].value_counts())
#fill_num_attrition=lambda x: 1 if x=="Yes" else 0

#type(fill_num_attrition)

df["num_attrition"]=df["Attrition"].apply(lambda x: 1 if x=="Yes" else 0)

df["num_attrition"].value_counts()
df_cov=df.cov()

df_cov
# Importing necessary package for creating model

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics
# one hot encoding num_attrition

cat_col_rm_tgt=cat_col[1:]

num_col=df.select_dtypes(include=np.number).columns

one_hot=pd.get_dummies(df[cat_col_rm_tgt])

emp_atr_df=pd.concat([df[num_col],one_hot],axis=1)

emp_atr_df.head(10)
X=emp_atr_df.drop(columns=['num_attrition'])

y=emp_atr_df[['num_attrition']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
train_Pred = logreg.predict(X_train)
metrics.confusion_matrix(y_train,train_Pred)
metrics.accuracy_score(y_train,train_Pred)
test_Pred = logreg.predict(X_test)
metrics.confusion_matrix(y_test,test_Pred)
metrics.accuracy_score(y_test,test_Pred)
from sklearn.metrics import classification_report

print(classification_report(y_test, test_Pred))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

#from sklearn.cr import train_test_split
X_train, X_test, y_train, y_test = train_test_split(

X, y, test_size = 0.3, random_state = 100)

y_train=np.ravel(y_train)

y_test=np.ravel(y_test)

#y_train = y_train.ravel()

#y_test = y_test.ravel()
for K in range(25):

    K_value = K+1

    neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')

    neigh.fit(X_train, y_train) 

    y_pred = neigh.predict(X_test)

    print ("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:",K_value)