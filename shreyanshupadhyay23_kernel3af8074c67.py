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
data=pd.read_csv('/kaggle/input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv')
data
data.head()
data.corr()
data.isnull().sum()
data.describe()
import matplotlib.pyplot as plt

plt.boxplot(data['Cumulative number of case(s)'])
plt.boxplot(data['Number of deaths'])
plt.boxplot(data['Number recovered'])
data.columns
import sklearn

from sklearn.preprocessing import StandardScaler,scale,normalize

x = data[['Cumulative number of case(s)']]

y = data['Number of deaths']
from sklearn import preprocessing

x_array = np.array(data['Cumulative number of case(s)'])

normalized_X = preprocessing.normalize([x_array])

y_array = np.array(data['Number recovered'])

normalized_Y = preprocessing.normalize([y_array])
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(xtrain,ytrain)
pred=model.predict(xtest)
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,classification_report
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error,accuracy_score
confusion_matrix(ytest,pred)
accuracy_score(ytest,pred)
print(classification_report(ytest,pred))
mean_squared_error(ytest,pred)
mean_absolute_error(ytest,pred)
mean_squared_log_error(ytest,pred)