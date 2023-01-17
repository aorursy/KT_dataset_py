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

import matplotlib.pyplot as plt

import seaborn as sns

df=pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
df.head()
df.shape
df[['Amount','Time','Class']].describe()
df.columns
df.isna().any()
nfcount=0

notfraud =df['Class']

for i in range(len(notfraud)):

    if notfraud[i]==0:

        nfcount=nfcount+1



nfcount

per_nf=(nfcount/len(notfraud))*100

print(per_nf)
fcount=0

fraud =df['Class']

for i in range(len(fraud)):

    if fraud[i]==1:

        fcount=fcount+1



nfcount

per_f=(fcount/len(fraud))*100

print(per_f)
plot_data=pd.DataFrame()

plot_data['Fraud Transaction']=fraud

plot_data['Genuine Transaction']= notfraud

plot_data
plt.title("bar  plot fro fraud and genuine transaction")

sns.barplot(x='Fraud Transaction' , y='Genuine Transaction',data=plot_data,palette='Blues')
x=df['Amount']

y=df['Time']

plt.plot(x,y)

plt.title("Time vs amount")
plt.figure(figsize=(10,8),)

plt.title('Amount Distribution')



sns.distplot(df['Amount'],color='red')
fig, ax= plt.subplots(figsize=(16,8))

ax.scatter(df['Amount'],df['Time'])

ax.set_xlabel('Amount')

ax.set_ylabel('Time')

plt.show()
# correlational matrices

cor_mat=df.corr()

fig=plt.figure(figsize=(14,9))

sns.heatmap(cor_mat,vmax=0.9,square=True)

plt.show()
x=df.drop(['Class'],axis=1)

y=df['Class']

from sklearn.model_selection import train_test_split

xtrain,xtest , ytrain , ytest=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LogisticRegression

logis=LogisticRegression()

logis.fit(xtrain,ytrain)
y_pred=logis.predict(xtest)
from sklearn import metrics



cm=metrics.confusion_matrix(ytest,y_pred)

print(cm)
accuracy = logis.score(xtest,ytest)

print('Accuracy of the Logistic regression model :',accuracy*100,'%')