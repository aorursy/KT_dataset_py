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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
flight = pd.read_csv('/kaggle/input/flight-delay-prediction/Jan_2019_ontime.csv')

flight.head()
sns.set_style('whitegrid')

g = sns.FacetGrid(flight,hue="ARR_DEL15",palette='coolwarm',size=6,aspect=2)

g = g.map(plt.hist,'DISTANCE',bins=40,alpha=0.7)
flight[flight['DIVERTED']==1]
flight[flight['CANCELLED']==1]
flight_science = flight.drop(['OP_UNIQUE_CARRIER','OP_CARRIER','TAIL_NUM','ORIGIN','DEST','DEP_TIME_BLK',

                              'Unnamed: 21','DEP_TIME','ARR_TIME','DEST_AIRPORT_SEQ_ID','ORIGIN_AIRPORT_SEQ_ID'],axis=1)

flight_science=flight_science.dropna()

flight_science.head()
flight_science['OP_CARRIER_AIRLINE_ID'].nunique()

flight_science[(flight_science['ARR_DEL15']!=1.0)&(flight_science['ARR_DEL15']!=0.0)]
from sklearn.model_selection import train_test_split

x_train = flight_science.drop('ARR_DEL15',axis=1)

y_train = flight_science['ARR_DEL15']

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(verbose=3)

model.fit(x_train,y_train)
flight_learn = pd.read_csv('/kaggle/input/flight-delay-prediction/Jan_2020_ontime.csv')

flight_learn.head()
flight_science2 = flight_learn.drop(['OP_UNIQUE_CARRIER','OP_CARRIER','TAIL_NUM','ORIGIN','DEST','DEP_TIME_BLK',

                              'Unnamed: 21','DEP_TIME','ARR_TIME','DEST_AIRPORT_SEQ_ID','ORIGIN_AIRPORT_SEQ_ID'],axis=1)

flight_science2=flight_science2.dropna()

flight_science2.head()
x_test=flight_science2.drop('ARR_DEL15',axis=1)

y_test = flight_science2['ARR_DEL15']

prediction = model.predict(x_test)

from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,prediction))

print('\n')

print(classification_report(y_test,prediction))