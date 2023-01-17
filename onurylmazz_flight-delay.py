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

import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)

warnings.filterwarnings("ignore",category=FutureWarning)

from datetime import datetime

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
df=pd.read_csv("/kaggle/input/flightdata.csv")   #veri setimizi ekliyoruz

df.shape
df.isnull().values.any()
df.isnull().sum()
df.drop('Unnamed: 25',axis=1,inplace=True)    #bu sütun tamamen kayıp veriden oluşuyor. o yüzden silebiliriz

df.head()
df = df[["MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "ORIGIN", "DEST", "CRS_DEP_TIME", "ARR_DEL15"]]

df.head()
df.isnull().sum()
df = df.fillna({'ARR_DEL15': 1})     #nan değerleri modelimizde kullanabilmek için 1 e çeviriyoruz

df.isnull().sum()
df.describe().T
df=pd.get_dummies(df, columns=['ORIGIN', 'DEST'])    #modelimizde kullanabilmek için orıgın ve dest kolonları ekliyoruz

df.head()
df['CRS_DEP_TIME']=df['CRS_DEP_TIME']//100

df.head()
x=df.drop('ARR_DEL15',axis=1)    #bağımlı ve bağımsız değişkenimizi ayırıyoruz

y=df['ARR_DEL15']
y.value_counts()
y.value_counts().plot(kind='bar', title='Count (ARR_DEL15)');
X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.25, random_state=42 )
from lightgbm import LGBMClassifier

from sklearn.metrics import confusion_matrix   

from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight

class_weights = dict(zip(np.unique(y_train), class_weight.compute_class_weight('balanced',

                                                 np.unique(y_train),

                                                 y_train)))
lgbm_model=LGBMClassifier(class_weight=class_weights).fit(X_train,y_train)  
y_pred=lgbm_model.predict(X_test) 
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
from sklearn.metrics import confusion_matrix    

confusion_matrix(y_test,y_pred)
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve



y_proba= lgbm_model.predict_proba(X_test)

roc_auc_score(y_test, y_proba[:, 1])
fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])

plt.plot(fpr, tpr)

plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate');
def possibility_delay(date_time,origin,dest):

    

    flight_date = datetime.strptime(date_time, '%d/%m/%Y %H:%M:%S')

    month = flight_date.month

    day = flight_date.day

    day_of_week = flight_date.isoweekday()

    hour = flight_date.hour

    

    origin = origin.upper()

    dest = dest.upper()

    

    new_data = [{'MONTH': month,

              'DAY': day,

              'DAY_OF_WEEK': day_of_week,

              'CRS_DEP_TIME': hour,

              'ORIGIN_ATL': 1 if origin == 'ATL' else 0,

              'ORIGIN_DTW': 1 if origin == 'DTW' else 0,

              'ORIGIN_JFK': 1 if origin == 'JFK' else 0,

              'ORIGIN_MSP': 1 if origin == 'MSP' else 0,

              'ORIGIN_SEA': 1 if origin == 'SEA' else 0,

              'DEST_ATL': 1 if dest == 'ATL' else 0,

              'DEST_DTW': 1 if dest == 'DTW' else 0,

              'DEST_JFK': 1 if dest == 'JFK' else 0,

              'DEST_MSP': 1 if dest == 'MSP' else 0,

              'DEST_SEA': 1 if dest == 'SEA' else 0 }]



    return lgbm_model.predict_proba(pd.DataFrame(new_data))[0][0]
days = ('1. Day', '2.Day', '3.Day', '4.Day', '5.Day', '6.Day', '7.Day')

date_and_flight = (possibility_delay('1/7/2018 12:00:00', 'ATL', 'jfk'),

          possibility_delay('2/5/2016 20:00:00', 'DTW', 'SEA'),

          possibility_delay('2/6/2019 12:00:00', 'JFK', 'SEA'),

          possibility_delay('4/7/2017 12:00:00', 'MSP', 'ATL'),

          possibility_delay('22/10/2018 22:00:00', 'SEA', 'DTW'),

          possibility_delay('13/11/2019 17:00:00', 'DTW', 'MSP'),

          possibility_delay('9/12/2017 12:00:00', 'ATL', 'JFK'))



plt.bar(days, date_and_flight, align='center', alpha=0.5)

plt.ylabel('possibility of delay')

plt.ylim((0,1));