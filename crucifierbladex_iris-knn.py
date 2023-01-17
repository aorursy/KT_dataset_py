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
df=pd.read_csv('/kaggle/input/iris/Iris.csv')

df=df.drop(['Id'],axis=1)

df.head()

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

from sklearn.model_selection import train_test_split
x=df.drop(['Species'],axis=1)

y=df['Species'].values
x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.2)

x_train,y_train=scaler.fit_transform(x_train),scaler.fit_transform(y_train)
iner=[]

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

for k in range(1,40):

    model=KNeighborsClassifier(n_neighbors=k)

    model.fit(x_train,x_test)

    y_pred=model.predict(y_train)

    iner.append(np.mean(y_pred!=y_test))
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(12,10))

plt.style.use('ggplot')

sns.lineplot(x=range(1,40),y=iner)
model=KNeighborsClassifier(n_neighbors=10)

model.fit(x_train,x_test)

y_pred=model.predict(y_train)

from sklearn import metrics 

metrics.accuracy_score(y_test,y_pred)
metrics.confusion_matrix(y_test,y_pred)
print(metrics.classification_report(y_test,y_pred))
!pip install scikit-plot

import scikitplot as skplt

skplt.metrics.plot_roc_curve(y_test,model.predict_proba(y_train))
skplt.metrics.plot_precision_recall_curve(y_test,model.predict_proba(y_train))
sns.scatterplot(y_test,y_pred)