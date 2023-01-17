# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import statsmodels.api as sm

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset=pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
dataset.quality.unique()
dataset.isnull().sum()
dataset.describe()

sns.distplot(dataset['density'])



sns.distplot(dataset['sulphates']) #positive skewed
q=dataset['sulphates'].quantile(0.99)
dataset_a=dataset[dataset['sulphates']<q]
sns.distplot(dataset_a["sulphates"])

sns.distplot(dataset['pH'])#fine,no outliers


sns.distplot(dataset['alcohol'])
q=dataset_a['alcohol'].quantile(0.99)
dataset_b=dataset_a[dataset['alcohol']<q]
sns.distplot(dataset_b["alcohol"])

sns.distplot(dataset['quality'])


plt.scatter(dataset['sulphates'],dataset['quality'])
plt.scatter(dataset['alcohol'],dataset['quality'])
sns.distplot(dataset['fixed acidity'])
q=dataset_b['fixed acidity'].quantile(0.99)
dataset_c=dataset_b[dataset['alcohol']<q]
sns.distplot(dataset_c["alcohol"])


sns.distplot(dataset['volatile acidity'])
q=dataset_c['volatile acidity'].quantile(0.99)
dataset_d=dataset_c[dataset['volatile acidity']<q]
sns.distplot(dataset_d["volatile acidity"])



sns.distplot(dataset['citric acid'])
q=dataset_d['citric acid'].quantile(0.99)
dataset_e=dataset_d[dataset['citric acid']<q]
sns.distplot(dataset_e["citric acid"])

sns.distplot(dataset['residual sugar'])
q=dataset_e['residual sugar'].quantile(0.95)
dataset_f=dataset_e[dataset['residual sugar']<q]
sns.distplot(dataset_f["residual sugar"])

sns.distplot(dataset['chlorides'])
q=dataset_f['chlorides'].quantile(0.95)
dataset_g=dataset_f[dataset['chlorides']<q]
sns.distplot(dataset_g["chlorides"])


sns.distplot(dataset['free sulfur dioxide'])
q=dataset_g['free sulfur dioxide'].quantile(0.95)
dataset_h=dataset_g[dataset['free sulfur dioxide']<q]
sns.distplot(dataset_h["free sulfur dioxide"])

sns.distplot(dataset['total sulfur dioxide'])
q=dataset_h['total sulfur dioxide'].quantile(0.95)
dataset_i=dataset_h[dataset['total sulfur dioxide']<q]
sns.distplot(dataset_i["total sulfur dioxide"])




#check if the training data is balanced

sns.countplot(dataset_i['quality'])

#try converting the dependent variable into 2 classes ,1 for high quality and 0 for low quality
#new_data=dataset_i
dataset_i=dataset_i.reset_index()
dataset_i=dataset_i.drop('index',axis=1)
#Converting "Quality variable into 1 or 0 based on quality"
t=0
for values in dataset_i['quality']:
    if values>5:
         dataset_i['quality'][t]=1
       # print(new_data[X])
         t=t+1
    elif values>=0 and values<=5:
         dataset_i['quality'][t]=0
         t=t+1
    #elif values>=6 and values<=10:
         #new_data['quality'][t]=2
        # t=t+1
sns.countplot(dataset_i['quality'])
dataset_i.corr()
#dropping columns coz either they are correlated with eachother or are less correlated with output variable

dataset_i=dataset_i.drop('fixed acidity',axis=1)
dataset_i=dataset_i.drop('free sulfur dioxide',axis=1)
dataset_i=dataset_i.drop('residual sugar',axis=1)

Y=dataset_i['quality'].values
Y=pd.DataFrame(Y)
Y=Y.reset_index()
Y=Y.drop('index',axis=1)

X=dataset_i.drop('quality',axis=1)

X=X.reset_index()
X=X.drop('index',axis=1)
#dividing into test and train data(Optional coz u can directly test on the given data X and Y)
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
X_train=X_train.reset_index()
X_train=X_train.drop('index',axis=1)

Y_train=Y_train.reset_index()
Y_train=Y_train.drop('index',axis=1)
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X=scaler.fit_transform(X)

X=pd.DataFrame(X)
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X,Y)

log_pred=logreg.predict(X)

from sklearn.metrics import accuracy_score
accuracy_score(Y,log_pred)
from sklearn.ensemble import RandomForestClassifier 
rfc = RandomForestClassifier()
rfc.fit(X, Y)
#Prediction
predictions = rfc.predict(X)
predictions=pd.DataFrame(predictions)

from sklearn.metrics import accuracy_score
accuracy_score(Y, predictions)
from sklearn.metrics import confusion_matrix

confusion_matrix(Y,predictions)
from sklearn.metrics import precision_score, recall_score
precision_score(Y,predictions,average='micro')
