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
df=pd.read_csv('/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv')
df.head()
df.tail()
df.isnull().sum()
df=df.drop(['Unnamed: 0'],axis=1)

df.info()
from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()

df['Time Serie']=label.fit_transform(df['Time Serie'])

df['AUSTRALIA - AUSTRALIAN DOLLAR/US$']=label.fit_transform(df['AUSTRALIA - AUSTRALIAN DOLLAR/US$'])

df['EURO AREA - EURO/US$']=label.fit_transform(df['EURO AREA - EURO/US$'])

df['NEW ZEALAND - NEW ZELAND DOLLAR/US$']=label.fit_transform(df['NEW ZEALAND - NEW ZELAND DOLLAR/US$'])

df['UNITED KINGDOM - UNITED KINGDOM POUND/US$']=label.fit_transform(df['UNITED KINGDOM - UNITED KINGDOM POUND/US$'])

df['BRAZIL - REAL/US$']=label.fit_transform(df['BRAZIL - REAL/US$'])

df['CANADA - CANADIAN DOLLAR/US$']=label.fit_transform(df['CANADA - CANADIAN DOLLAR/US$'])

df['CHINA - YUAN/US$']=label.fit_transform(df['CHINA - YUAN/US$'])

df['HONG KONG - HONG KONG DOLLAR/US$']=label.fit_transform(df['HONG KONG - HONG KONG DOLLAR/US$'])

df['INDIA - INDIAN RUPEE/US$']=label.fit_transform(df['INDIA - INDIAN RUPEE/US$'])



df['KOREA - WON/US$']=label.fit_transform(df['KOREA - WON/US$'])

df['MEXICO - MEXICAN PESO/US$']=label.fit_transform(df['MEXICO - MEXICAN PESO/US$'])

df['SOUTH AFRICA - RAND/US$']=label.fit_transform(df['SOUTH AFRICA - RAND/US$'])

df['SINGAPORE - SINGAPORE DOLLAR/US$']=label.fit_transform(df['SINGAPORE - SINGAPORE DOLLAR/US$'])

df['DENMARK - DANISH KRONE/US$']=label.fit_transform(df['DENMARK - DANISH KRONE/US$'])

df['JAPAN - YEN/US$']=label.fit_transform(df['JAPAN - YEN/US$'])

df['MALAYSIA - RINGGIT/US$']=label.fit_transform(df['MALAYSIA - RINGGIT/US$'])

df['NORWAY - NORWEGIAN KRONE/US$']=label.fit_transform(df['NORWAY - NORWEGIAN KRONE/US$'])

df['SWEDEN - KRONA/US$']=label.fit_transform(df['SWEDEN - KRONA/US$'])

df['SRI LANKA - SRI LANKAN RUPEE/US$']=label.fit_transform(df['SRI LANKA - SRI LANKAN RUPEE/US$'])



df['SWITZERLAND - FRANC/US$']=label.fit_transform(df['SWITZERLAND - FRANC/US$'])

df['TAIWAN - NEW TAIWAN DOLLAR/US$']=label.fit_transform(df['TAIWAN - NEW TAIWAN DOLLAR/US$'])

df['THAILAND - BAHT/US$']=label.fit_transform(df['THAILAND - BAHT/US$'])
df.info()
import matplotlib.pyplot as plt

import seaborn as sns

fig= plt.figure(figsize=(30,20))

sns.countplot('Time Serie',data=df)

plt.title('Time Serie or Date')

plt.ylabel("Time Serie")

plt.show()
from sklearn.model_selection import train_test_split

train,test=train_test_split(df,test_size=0.1,random_state=1)
def data_split(df):

    x=df.drop(['INDIA - INDIAN RUPEE/US$'],axis=1)

    y=df['INDIA - INDIAN RUPEE/US$']

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