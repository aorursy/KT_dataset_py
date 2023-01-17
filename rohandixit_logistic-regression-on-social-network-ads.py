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
import matplotlib.pyplot as plt
import sklearn as sns
data = pd.read_csv("/kaggle/input/social-network-ads/Social_Network_Ads.csv")
data.head()
# Here User ID is not suitable to predict the results,so we are ignore this coloumn.

data = data[['Gender','Age','EstimatedSalary','Purchased']]
print(data.head())
data.isnull().sum()
data.plot()
import seaborn as sns
sns.set(style="ticks")

sns.pairplot(data, hue="Gender")
# Here We Check The Total No. Who Purchased or Not Purchased
sns.countplot(x="Purchased",data=data)
# As We See here mostly female's like to buy product then male's
sns.countplot(x="Purchased",hue="Gender",data=data)
# Now Lets Convert The Variables into dummy variabels for our ML model.
# If the Value of 1 in Male Then i.e male if value is 1 in Female then i.e Female 
pd.get_dummies(data['Gender'])
sex = pd.get_dummies(data['Gender'],drop_first=True)
sex.head()
data_p = pd.concat([data,sex],axis=1)
data_p.head()
# Now There is a Gender Column which we do neet further because we have converted into dummies and concat the male column in data set
data_p = data_p.drop(['Gender'],axis=1)
data_p.head()
X = data_p[['Age','EstimatedSalary','Male']].values
y = data_p['Purchased'].values
# Now Train our Data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.3)
# Preprocessing
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.fit_transform(X_test)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
predict = log.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predict))
print(classification_report(y_test,predict))
plt.plot(y_test,predict)
plt.show()
