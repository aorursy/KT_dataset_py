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
df=pd.read_csv('/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv')
df.head()
df.tail()
df.isnull().sum()
df=df.drop(['Unnamed: 0','title_status',],axis=1)

df.info()
from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()

df['brand']=label.fit_transform(df['brand'])

df['model']=label.fit_transform(df['model'])

df['color']=label.fit_transform(df['color'])

df['vin']=label.fit_transform(df['vin'])

df['state']=label.fit_transform(df['state'])

df['country']=label.fit_transform(df['country'])

df['condition']=label.fit_transform(df['condition'])



df.info()
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot('brand',data=df)

plt.title('Brand List')

plt.ylabel("brand")

plt.show()
sns.scatterplot(x=df.model,y=df.mileage)
sns.distplot(df.condition.dropna(),bins=8,color='violet')
from sklearn.model_selection import train_test_split

train,test=train_test_split(df,test_size=0.1,random_state=1)
def data_split(df):

    x=df.drop(['condition'],axis=1)

    y=df['condition']

    return x,y



x_train,y_train=data_split(train)

x_test,y_test=data_split(test)
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression



log_model=LogisticRegression()

log_model.fit(x_train,y_train)

prediction=log_model.predict(x_test)

score=accuracy_score(y_test,prediction)

print (score)
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

smote=XGBClassifier()

smote.fit(x_train,y_train)



smote_pred=smote.predict(x_test)

accuracy=accuracy_score(y_test,smote_pred)

print("Test accuracy is {:.2f}%".format(accuracy*100.0))