# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from sklearn import preprocessing



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/sloan-digital-sky-survey/Skyserver_SQL2_27_2018 6_51_39 PM.csv")
df.head(10)
df.columns
labels = df['class']

df.drop('class' , inplace = True , axis = 1 )

print(labels.unique())
listt = df.columns

dict = []

for i in listt :

    dict.append(( len(df[i].unique()) / df.shape[0] , i ))
dict = sorted(dict)

for i , j in   dict:

    print( i , j )
# drop first 2 columns 

df.drop(['objid' , 'rerun'] , inplace = True , axis = 1 )
dict2 = []

for i in df.columns :

    dict2.append( ( df[i].isnull().sum() , i ) )

dict2 = sorted(dict2 , reverse = False)

for i , j in dict2 :

    print( i , j )
df.describe()
from sklearn.preprocessing import StandardScaler



ss = StandardScaler()

df = ss.fit_transform(df)

df1 = pd.DataFrame(df)
le = preprocessing.LabelEncoder()

labels = le.fit_transform(labels)

print(labels)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df1, labels, test_size=0.33)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn import svm

clf = svm.SVC(kernel='rbf')

clf.fit(X_train, y_train)
print(clf.score(X_test , y_test))

from sklearn.metrics import f1_score



y_pred = clf.predict(X_test)

print(f1_score(y_test, y_pred, average='macro'))

print(f1_score(y_test, y_pred, average='micro'))

print(f1_score(y_test, y_pred, average='weighted'))

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

clf.fit(X_train,y_train)

print(clf.score(X_test , y_test))

from sklearn.metrics import f1_score



y_pred = clf.predict(X_test)

print(f1_score(y_test, y_pred, average='macro'))

print(f1_score(y_test, y_pred, average='micro'))

print(f1_score(y_test, y_pred, average='weighted'))

label_df = pd.DataFrame(y_train)

zeros = label_df[label_df == 0 ].count()

ones  =  label_df[label_df == 1 ].count()

two = label_df[label_df == 2 ].count()
import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt



objects = ('0', '1', '2' )



y_pos = np.arange(len(objects))



performance = [int(zeros) , int(ones) , int(two)]



plt.bar(y_pos, performance, align='center', alpha=0.5)



plt.xticks(y_pos, objects)



plt.ylabel('Usage')



plt.title('Programming language usage')



plt.show()
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

clf.fit(X_train, y_train)
from sklearn.metrics import f1_score

print(clf.score(X_test  , y_test))

y_pred = clf.predict(X_test)

print(f1_score(y_test, y_pred, average='macro'))

print(f1_score(y_test, y_pred, average='micro'))

print(f1_score(y_test, y_pred, average='weighted'))
