# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/No-show-Issue-Comma-300k.csv')

df.head()
df.AppointmentRegistration=df.AppointmentRegistration.apply(np.datetime64)

df.ApointmentData=df.ApointmentData.apply(np.datetime64)

df.AwaitingTime=df.AwaitingTime.apply(abs)
df['gap_time']=(df.ApointmentData-df.AppointmentRegistration).apply(lambda x:x.total_seconds()/(3600*24))
def chang_show(data):

    if data=='Show-Up':

        return 1

    else:

        return 0
df['Status']=df.Status.apply(chang_show)
df=pd.get_dummies(df)

df
y=df.Status.as_matrix()

df_copy=df.drop(['Status','AppointmentRegistration','ApointmentData','AwaitingTime'],axis=1)

x=df_copy.as_matrix()
from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y)
from sklearn.linear_model import LogisticRegression

clf=LogisticRegression()

clf.fit(x_train,y_train)
from sklearn.metrics import accuracy_score

print('Accuracy:',round(accuracy_score(y_test,clf.predict(x_test)),2)*100,'%')
clf.coef_
df_copy.columns 
feat=['Alcoolism','Smokes',

       'Scholarship', 'Tuberculosis', 'Sms_Reminder', 'AwaitingTime',

       'gap_time', 'Gender_F', 'Gender_M', 'DayOfTheWeek_Friday','DayOfTheWeek_Thursday', 'DayOfTheWeek_Tuesday',

       'DayOfTheWeek_Wednesday','DayOfTheWeek_Saturday']

x=df[feat].as_matrix()
x_train,x_test,y_train,y_test=train_test_split(x,y)

clf=LogisticRegression()

clf.fit(x_train,y_train)

print('Accuracy:',round(accuracy_score(y_test,clf.predict(x_test)),2)*100,'%')