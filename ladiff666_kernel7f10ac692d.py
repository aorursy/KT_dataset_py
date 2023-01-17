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
data = pd.read_csv('../input/ph-recognition/ph-data.csv')

data['type'] = 0

data['label'] = data['label'].astype('int')

data.loc[(data.label < 7) &  (data.label>4),'type'] = 1

data.loc[data.label < 4,'type'] = 2

data.loc[data.label == 7,'type'] = 3

data.loc[(data.label > 7) & (data.label<=10),'type'] = 4

data.loc[data.label > 10,'type'] = 5

data

from sklearn.ensemble import RandomForestClassifier 

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import classification_report, accuracy_score, f1_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.pipeline import make_pipeline



X = data.drop('label',axis=1)

Y = data['label']

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.33)



for n in range(1,11):

    print('para=', n)

    pipe = make_pipeline(StandardScaler(),KNeighborsClassifier())

    pipe.fit(X_train,y_train)

    Y = pipe.predict(X_test)

    print(classification_report(y_true=y_test, y_pred=Y))

    print('--------------------------')