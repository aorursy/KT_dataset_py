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

from scipy.stats import zscore

from collections import Counter
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

df.head(3)
sns.countplot(df['Class'])

x = df.drop(columns = ['Class'])

y = df.Class
# Shape of the x and y

print(x.shape,y.shape)

x.info()

x['Amount'] = zscore(x['Amount'])

x['Time']= zscore(x['Time'])
plt.figure(figsize=(12,6))

sns.lmplot(x = 'Time',y='Amount',hue = 'Class',data = df,fit_reg=False)
from imblearn.under_sampling import NearMiss
nm = NearMiss()

x_bal,y_bal = nm.fit_sample(x,y)

print("x :-",x_bal.shape,"y :-",y_bal.shape)
frame = [x_bal,y_bal]

df_under = pd.concat(frame,axis=1)

sns.lmplot(x = 'Time', y = 'Amount',hue = 'Class',fit_reg=False,data = df_under)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
xtrain,xtest,ytrain,ytest =train_test_split(x_bal,y_bal,test_size = 0.2,random_state = 42)

lr = LogisticRegression()
lr.fit(xtrain,ytrain)

yp = lr.predict(xtest)
accuracy_score(ytest,yp)
sns.heatmap(confusion_matrix(ytest,yp,labels=[0,1]),annot=True)
from imblearn.over_sampling import RandomOverSampler
os = RandomOverSampler(random_state=42)

x_bal,y_bal = os.fit_sample(x,y)
frame = [x_bal,y_bal]

df_under = pd.concat(frame,axis=1)

sns.lmplot(x = 'Time', y = 'Amount',hue = 'Class',fit_reg=False,data = df_under)
x_bal.shape,y_bal.shape

xttrain,xtest,ytrain,ytest = train_test_split(x_bal,y_bal,test_size = 0.2,random_state = 42)

lr.fit(xttrain,ytrain)
y_pred = lr.predict(xtest)

accuracy_score(ytest,y_pred)


sns.heatmap(confusion_matrix(ytest,y_pred,labels=[0,1]),annot=True)
print(classification_report(ytest,y_pred))
from imblearn.combine import SMOTETomek
smk = SMOTETomek(random_state=42)

x_smk,y_smk = smk.fit_sample(x,y)
frame = [x_smk,y_smk]

df_under = pd.concat(frame,axis=1)

sns.lmplot(x = 'Time', y = 'Amount',hue = 'Class',fit_reg=False,data = df_under)
x_smk.shape,y_smk.shape
xttrain,xtest,ytrain,ytest = train_test_split(x_smk,y_smk,test_size = 0.2,random_state = 42)

lr.fit(xttrain,ytrain)
ypred = lr.predict(xtest)

accuracy_score(ytest,ypred)
sns.heatmap(confusion_matrix(ytest,ypred,labels=[0,1]),annot=True)
print(classification_report(ytest,ypred))