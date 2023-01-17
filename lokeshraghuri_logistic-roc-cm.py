# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

#import matplot.pyplot as plt

dataset = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

dataset

X = dataset.iloc[:,:-1]

y = dataset.iloc[:,8]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression

logis = LogisticRegression()

logis.fit(X_train,y_train)

y_pred = logis.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred,y_test)

cm

logis.score(X_test,y_test)

logis.score(X_train,y_train)

from sklearn.metrics import classification_report

report = classification_report(y_test,y_pred)

print(report)

from sklearn.metrics import confusion_matrix,classification_report,roc_curve,roc_auc_score

fpr,tpr,_= roc_curve(logis.predict(X),y,drop_intermediate=False)

import matplotlib.pyplot as plt

plt.figure()

plt.plot(fpr,tpr,color='red',Label='ROC_Curve')

plt.plot([0,1],[0,1],color='blue',linestyle='--')

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.title('ROC CURVE')

plt.show()


