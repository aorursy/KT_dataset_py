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
df_train= pd.read_csv("/kaggle/input/train.csv")

df_test= pd.read_csv("/kaggle/input/test.csv")
freq =df_train['condition'].dropna().mode()[0]

df_train['condition'] = df_train['condition'].fillna(freq)

df_test['condition'] = df_test['condition'].fillna(freq)
y_train=np.array(df_train.iloc[:,9:])

id_test=np.array(df_test.iloc[:,0])

x_train=np.array(df_train.iloc[:,3:9])

x_test=np.array(df_test.iloc[:,3:9])
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(np.unique(x_train[:,1]))

x_train[:,1]=le.transform(x_train[:,1])

x_test[:,1]=le.transform(x_test[:,1])
x_test
from sklearn.ensemble import RandomForestClassifier



rforest = RandomForestClassifier(n_estimators = 1000, random_state = 1) 

rforest.fit(x_train,y_train)

y_test = rforest.predict(x_test)
(id_test, y_test)
df3 = pd.DataFrame()

df3['pet_id'] = id_test.reshape(len(y_test)).tolist()

df3['breed_category'] = y_test[:,0].reshape(len(y_test)).tolist()

df3['pet_category'] = y_test[:,1].reshape(len(y_test)).tolist()
df3.to_csv("./file.csv", sep=',',index=True)