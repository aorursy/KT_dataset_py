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
records=pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

records.head()
records.info()
print("Max Age {:.2f}".format(max(records['age'])))

print("Min Age {:.2f}".format(min(records['age'])))
def normalize(dataset):

    data=(dataset-dataset.min())/(dataset.max()-dataset.min())

    data['DEATH_EVENT']=dataset['DEATH_EVENT']

    return data

record=normalize(records)

record.head()

    
record.describe()
record.corr()
from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import confusion_matrix,plot_confusion_matrix
sns.heatmap(record.corr(),vmin=-1,vmax=1,center=0,cmap=sns.diverging_palette(20,220,n=200))
sns.regplot(x='age',y='serum_creatinine',data=record)
sns.regplot(x='time',y='ejection_fraction',data=record)
sns.set(style='ticks')

sns.pairplot(record[:13])
Scaler=MinMaxScaler()

X=records[['age','anaemia','ejection_fraction','high_blood_pressure',

           'platelets','serum_creatinine','smoking','time']]

Y=records[['DEATH_EVENT']]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0)

X_train_scale=Scaler.fit_transform(X_train)

X_test_scale=Scaler.fit_transform(X_test)

rf=RandomForestClassifier(random_state=0,max_features=5).fit(X_train_scale,Y_train)

ran=rf.predict(X_test_scale)



ranf=rf.predict(X_test)
print('Accuracy Score for Random Forest',accuracy_score(Y_test,ran))

print('Precision Score {:.2f}'.format(precision_score(Y_test,ran)))

print('F1 Score {:.2f}'.format(f1_score(Y_test,ran)))

print('Recall Score {:.2f}'.format(recall_score(Y_test,ran)))
print(confusion_matrix(Y_test,ran))

plot_confusion_matrix(rf,X_test_scale,Y_test)
merge=pd.merge(X_test,df,how='inner',left_index=True,right_index=True)

merge.head()