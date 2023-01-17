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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('/kaggle/input/car-insurance-claim/file(3).csv')
pd.set_option('Display.max_columns' , 25)
df.head(10)
df.describe()
df.isna().sum()
df.dropna(axis = 'index' , how = 'any' , inplace = True)
df.drop(df[['KIDSDRIV' , 'AGE' , 'HOMEKIDS' , 'YOJ' , 'PARENT1' , 'HOME_VAL' , 'MSTATUS'
            , 'GENDER' , 'EDUCATION' , 'TIF' , 'RED_CAR']], axis =1 , inplace = True)
X = df.iloc[: ,1:-1 ]
Y = df.iloc[: , -1]
dummie1 = pd.get_dummies(df['OCCUPATION'])
dummie2 = pd.get_dummies(df['CAR_USE'])
dummie3 = pd.get_dummies(df['CAR_TYPE'])
dummies = [dummie1 , dummie2 , dummie3]
X.drop(['OCCUPATION', 'CAR_USE' , 'CAR_TYPE']  ,axis = 1, inplace = True)
X = pd.concat([X , dummie1] , axis = 1)
X = pd.concat([X , dummie2] , axis = 1)
X = pd.concat([X , dummie3] , axis = 1)
X['INCOME'] = X['INCOME'].str.replace('$' , '')
X['INCOME'] = X['INCOME'].str.replace(',' , '')
X['INCOME'] = X['INCOME'].astype(int)
X['BLUEBOOK'] = X['BLUEBOOK'].str.replace('$' , '')
X['BLUEBOOK'] = X['BLUEBOOK'].str.replace(',' , '')
X['BLUEBOOK'] = X['BLUEBOOK'].astype(int)
X['OLDCLAIM'] = X['OLDCLAIM'].str.replace('$' , '')
X['OLDCLAIM'] = X['OLDCLAIM'].str.replace(',' , '')
X['OLDCLAIM'] = X['OLDCLAIM'].astype(int)
X['CLM_AMT'] = X['CLM_AMT'].str.replace('$' , '')
X['CLM_AMT'] = X['CLM_AMT'].str.replace(',' , '')
X['CLM_AMT'] = X['CLM_AMT'].astype(int)
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , classification_report , accuracy_score
labelencoder = LabelEncoder()
onehotencoder = OneHotEncoder()
log_reg = LogisticRegression()
X['REVOKED'] = labelencoder.fit_transform(X['REVOKED'])
X_train , X_test , Y_train , Y_test = train_test_split(X,Y , test_size = 0.2 , random_state = 1)
log_reg.fit(X_train , Y_train)
predictions = log_reg.predict(X_test)
classification_report(Y_test , predictions)
confusion_matrix(Y_test , predictions)
accuracy_score(Y_test , predictions)