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

#using logistc regression and nearmiss



#model was very very accurate

%matplotlib inline

import matplotlib.pyplot as plt



from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss

import seaborn as sns

from pandas import DataFrame

from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score, precision_score, confusion_matrix

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split

#from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import auc, roc_curve,roc_auc_score



df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

corre = df.corr()

#print(df.corr()["Class"].sort_values())

#sns.heatmap(corre)

df.drop(['V17','V14','V12','V10','V16','V3','V7','V18','V1','V9','V5','V6','Time','Amount'], axis=1,inplace=True)



X,Y = df.iloc[:,:-1],df.iloc[:,-1]

stdscale = MinMaxScaler()

X = pd.DataFrame(stdscale.fit_transform(X), columns = X.columns)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3, random_state=2)



sm = NearMiss()



X_train, Y_train= sm.fit_sample(X_train,Y_train)



print(sum(Y_train==0))



lgf = LogisticRegression(random_state=2)



lgf.fit(X_train,Y_train)

y_predict = lgf.predict(X_test)

print(classification_report(y_predict,Y_test))

fpr, tpr, thresholds = roc_curve(Y_test, y_predict)

print(roc_auc_score(Y_test,y_predict))

plt.plot(fpr,tpr,linewidth=3)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('false rate')

plt.ylabel('true rate')

plt.title('ROC CURVE')

plt.show()