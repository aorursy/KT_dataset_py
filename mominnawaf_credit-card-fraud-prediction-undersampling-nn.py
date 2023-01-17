# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier #Multi-Layerd Precptron Classifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import recall_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.info()
df=df.drop(['Time'],axis=1)
count_class=pd.value_counts(df['Class'],sort=False)
count_class.plot(kind='bar')

plt.xlabel="Class"

plt.ylabel="Frequcency"

plt.show()
df['NormAmount']=StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
x=df.iloc[:,df.columns!="Class"]

y=df.iloc[:,df.columns=='Class']

len(y[y.Class==1])
number_of_fraud=len(df[df.Class==1])

fraud_index=np.array(df[df.Class==1].index)

normal_index=np.array(df[df.Class==0].index)

random_index=np.random.choice(normal_index,number_of_fraud,replace=False)

undersample_index=np.random.choice(fraud_index,number_of_fraud)
undersample_data=df.iloc[undersample_index,:]

X=undersample_data.iloc[:,undersample_data.columns!='Class']

Y=undersample_data.iloc[:,undersample_data.columns=='Class']
xtrain,xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3)
xtrain_under,xtest_under,ytrain_under,ytest_under=train_test_split(X,Y,test_size=0.3,random_state=0)
MLPC=MLPClassifier(hidden_layer_sizes=(100,),max_iter=1000)

MLPC.fit(xtrain_under,ytrain_under)

y_predict=MLPC.predict(xtest)

recall_acc = recall_score (ytest,y_predict)

recall_acc 