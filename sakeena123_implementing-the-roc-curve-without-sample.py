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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df=pd.read_csv('../input/creditcard.csv')
df.describe()
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df['Time']=sc.fit_transform(df['Time'].values.reshape(-1,1))
df['Amount']=sc.fit_transform(df['Amount'].values.reshape(-1,1))
x=df.iloc[:,:-1].values
y=df.iloc[:,30].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=2/3,random_state=0)
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(x_train,y_train)
y_pred=log.predict(x_test)
y_pred
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,auc
cm=confusion_matrix(y_test,y_pred)
cm
print(classification_report(y_test,y_pred))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc=auc(false_positive_rate, true_positive_rate)
plt.figure()
lw = 2
plt.plot(false_positive_rate, true_positive_rate, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

