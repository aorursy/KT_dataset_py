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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

from datetime import timedelta

import warnings

warnings.filterwarnings("ignore")



df=pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")



df.head()





df["country"]=df["country"].fillna(0)

df["children"]=df["children"].fillna(df["children"].mode()[0])


def arrival_date(data):

    month=data["arrival_date_month"]

    year=data["arrival_date_year"]

    day_of_month=data["arrival_date_day_of_month"]

    date=datetime.datetime.strptime(str(year)+" "+month+" "+str(day_of_month),"%Y %B %d")

    return date

def booking_date(data):

    return data["arrival_date"]-timedelta(data["lead_time"])

def family_or_not(data):

    adult=data["adults"]

    child=data["children"]

    baby=data["babies"]

    if (adult>0) &((child>0)|(baby>0)):

        return 1

    else:

        return 0

def room_type_not_given(data):

    reserved_room=data["reserved_room_type"]

    assigned_room=data["assigned_room_type"]

    if reserved_room==assigned_room:

        return 0

    else: 

        return 1
def feature_eng(data):

    data["arrival_date"]=data.apply(arrival_date,axis=1)

    data["arrival_date_weekday"]=data["arrival_date"].dt.weekday

    data["booking_date"]=data.apply(booking_date,axis=1)

    data["Total_No_Of_Nights_Stayed"]=data["stays_in_weekend_nights"]+data["stays_in_week_nights"]

    data["family_or_not"]=data.apply(family_or_not,axis=1)

    data["total_no_of_people"]=data["adults"]+data["children"]+data["babies"]

    data["room_type_changes"]=data.apply(room_type_not_given,axis=1)

    data["Non_Refund_Flag"]=data["deposit_type"].apply(lambda x:1 if x=="Non Refund" else 0)

    data["days_in_waiting_list_flag"]=data["days_in_waiting_list"].apply(lambda x:0 if x==0 else 1)

    data["adr"]=data["adr"].apply(lambda x:x if x>=0 else data["adr"].median())

    data["missing_features"]=(data==0).sum(axis=1)

    return data
df1=feature_eng(df)
df2=df1.drop(columns=["agent","company","reservation_status_date","arrival_date","booking_date","reservation_status"])
df2.columns
df_dummies=pd.get_dummies(data=df2,drop_first=True)
y=df_dummies["is_canceled"]

X=df_dummies.drop("is_canceled",axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.metrics import confusion_matrix,classification_report,recall_score,roc_auc_score
def model_eval(algo,Xtrain,ytrain,Xtest,ytest):

    algo.fit(Xtrain,ytrain)



    y_train_pred=algo.predict(Xtrain)

    y_train_prob=algo.predict_proba(Xtrain)[:,1]

    from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,classification_report,confusion_matrix

    y_test_pred=algo.predict(Xtest)

    y_test_prob=algo.predict_proba(Xtest)[:,1]

    print(algo)

    print('\n'*1)

    print(' Accuracy score of train: ', accuracy_score(ytrain,y_train_pred))

    print(' Accuracy score of test: ', accuracy_score(ytest,y_test_pred))

    print('\n'*1)

    print(' Confusion Matrix of train: ', confusion_matrix(ytrain,y_train_pred))

    print(' Confusion Matrix of test: ', confusion_matrix(ytest,y_test_pred))

    print('\n'*1)

    print(' Auc of train: ', roc_auc_score(ytrain,y_train_prob))

    print(' Auc of test: ', roc_auc_score(ytest,y_test_prob))
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(max_depth=10)
model_eval(dtc,X_train,y_train,X_test,y_test)
pd.DataFrame(index=X.columns,data=dtc.feature_importances_,columns=["Scores"]).sort_values("Scores",ascending=False)[:15].plot.bar()