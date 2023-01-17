# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import numpy as np

import  matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#loading the data

creditfraud = pd.read_csv("../input/creditcard.csv")

pd.set_option('display.max_columns', 500)
creditfraud.describe()
creditfraud.head()
creditfraud["V1"].isnull().sum()
col_list = ["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"]
j = 0

for i in col_list:

    p=creditfraud[i].isnull().sum()

    j=j+1

    print(j,p)


from sklearn.model_selection import train_test_split


X_col_list = ["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"]




X = creditfraud[X_col_list]






Y = creditfraud["Class"]


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


lr = LogisticRegression()
lr.fit(X_train,y_train)
lr.score(X_train,y_train)
pre=lr.predict(X_test)
arr = np.array(y_test)


j=0

for i in range(0,len(pre)):

   if pre[i]!=  arr[i]:

    j = j+1

    
#% of accuracy

acc = ((len(pre)-j)/len(pre))*100


acc
pre = np.array(pre)
import math






#root mean squre error

rms = math.sqrt((sum((arr-pre)**2))/len(pre))
#rms  accuracy %

rms_acc_per=100-rms*100
rms_acc_per


#count_classes = pd.value_counts(creditfraud['Class'], sort = True).sort_index()

#count_classes.plot(kind = 'bar')

sns.countplot(x="Class",data=creditfraud)

plt.title("Fraud class histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")
y_test=pd.DataFrame(y_test)

sns.countplot(x="Class",data=y_test)

#count_classes = pd.value_counts(y_test, sort = True).sort_index()

#count_classes.plot(kind = 'bar')

plt.title("Actual Fraud class histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")


pre.columns = ['Class']

pre = pd.DataFrame(pre)

sns.countplot(x="Class",data=pre)

#count_classes = pd.value_counts(pre, sort = True).sort_index()

#count_classes.plot(kind = 'bar')

plt.title("Predicted Fraud class histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")

plt.show()
#importing the f1_score

from sklearn.metrics import f1_score
f_one_score = f1_score(y_test,pre,average='macro')
f_one_score
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)
rf.score(X_train,y_train)
y_test=pd.DataFrame(y_test)

sns.countplot(x="Class",data=y_test)

#count_classes = pd.value_counts(y_test, sort = True).sort_index()

#count_classes.plot(kind = 'bar')

plt.title("Actual Fraud class histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")
rf_pre = rf.predict(X_test)

rf_pre=pd.DataFrame(rf_pre)

rf_pre.columns = ["Class"]
rf_pre=pd.DataFrame(rf_pre)

sns.countplot(x="Class",data=rf_pre)

#count_classes = pd.value_counts(y_test, sort = True).sort_index()

#count_classes.plot(kind = 'bar')

plt.title("Predicted Fraud class histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")
f_one_score_rf = f1_score(y_test,rf_pre,average='macro')
f_one_score_rf