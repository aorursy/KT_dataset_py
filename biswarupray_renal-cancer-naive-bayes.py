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
import pandas as pd

df = pd.read_csv("../input/renal-cancer/Merge 72 csv.csv")
df.head()
import pandas as pd

import numpy as np

from sklearn import preprocessing 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()

  

#Check for the missing values in the columns 

fig, ax = plt.subplots(figsize=(15,9))

sns.heatmap(df.isnull(), cbar=False, cmap="YlGnBu_r")

plt.show()
#dropping the first two rows

indexes_to_keep = set(range(df.shape[0])) - set([0,1])

df = df.take(list(indexes_to_keep))

df.head()
df_train=df.drop(["ajcc_pathologic_tumor_stage","bcr_patient_uuid","bcr_patient_barcode"],axis=1)

df_train=pd.get_dummies(df_train)

df_train.head()
x=df_train.iloc[:,1:-1].values

x
from sklearn.preprocessing import LabelEncoder



labelencoder = LabelEncoder()

df['ajcc_pathologic_tumor_stage'] = labelencoder.fit_transform(df['ajcc_pathologic_tumor_stage'])

df['ajcc_pathologic_tumor_stage']
y=df.iloc[:,72:73].values

y
from statistics import mean 

#i=0

count=0.15

z=0

acc=[]

ts=[]

#list1=[]

while(count<0.9):

    list1=[]

    i=0

    sum=0.0

    while(i<50):

        from sklearn.model_selection import train_test_split

        #Split dataset into training set and test set

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=count)



        from sklearn.preprocessing import StandardScaler

        scaler=StandardScaler()

        scaler.fit(x_train)

        x_train=scaler.transform(x_train)

        x_test=scaler.transform(x_test)





        from sklearn.naive_bayes import GaussianNB

        gnb=GaussianNB()

        gnb.fit(x_train,y_train.ravel())

        y_pred=gnb.predict(x_test)



        from sklearn import metrics

       # Model Accuracy

        z=(metrics.accuracy_score(y_test, y_pred))

        i=i+1

        sum=sum+z

      

    acc.append(sum/50)

    ts.append(round(count,2))

    sum=0.0

    #list1=[]

    count=count+0.05

print(ts)

print(acc)

len(acc)
import matplotlib.pyplot as plt

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

width = 0.3

ax.scatter(ts,acc)

ax.set_xlabel('test size')

ax.set_ylabel('accuracy')

plt.title("Naive Bayes for 72")

plt.show()