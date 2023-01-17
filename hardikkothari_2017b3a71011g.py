# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/minor-project-2020/train.csv')

#df.shape

x=df.iloc[:,1:89]

y=df.iloc[:,-1]

df.target.value_counts()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=121)

from imblearn.over_sampling import SMOTE



resm = SMOTE(random_state=27, sampling_strategy=1.0)

X_train, y_train = resm.fit_sample(X_train, y_train)
from sklearn.preprocessing import StandardScaler



scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)

X_test = scalar.transform(X_test)
from sklearn.feature_selection import RFECV

from sklearn.tree import DecisionTreeClassifier



#clf = DecisionTreeClassifier()

#trans = RFECV(clf)

#new_trans = trans.fit_transform(X_train,y_train)

#columns_retained_RFECE = df.iloc[ : ,1 : 89].columns[trans.get_support()].values
#columns_retained_RFECE#

#print(len(columns_retained_RFECE))
x_new=df[['col_0', 'col_1', 'col_2', 'col_3', 'col_4', 'col_6', 'col_9', 'col_16',

       'col_21', 'col_26', 'col_49', 'col_51', 'col_52', 'col_54', 'col_55',

       'col_56', 'col_57', 'col_58', 'col_59', 'col_60', 'col_61', 'col_62',

       'col_63', 'col_64', 'col_65', 'col_66', 'col_67', 'col_68', 'col_69',

       'col_70', 'col_71', 'col_72', 'col_73', 'col_75', 'col_76', 'col_77',

       'col_78', 'col_80', 'col_81', 'col_83', 'col_84', 'col_86']]
X_train, X_test, y_train, y_test = train_test_split(x_new, y, test_size=0.2, random_state=121)

scalar = StandardScaler()



X_train = scalar.fit_transform(X_train)

X_test = scalar.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

from sklearn.metrics import roc_curve, auc



FPR, TPR, _ = roc_curve( y_test, y_pred)

ROC_AUC = auc(FPR, TPR)

print(ROC_AUC)
df2 = pd.read_csv('/kaggle/input/minor-project-2020/test.csv',sep=',')
x=df2[['col_0', 'col_1', 'col_2', 'col_3', 'col_4', 'col_6', 'col_9', 'col_16',

       'col_21', 'col_26', 'col_49', 'col_51', 'col_52', 'col_54', 'col_55',

       'col_56', 'col_57', 'col_58', 'col_59', 'col_60', 'col_61', 'col_62',

       'col_63', 'col_64', 'col_65', 'col_66', 'col_67', 'col_68', 'col_69',

       'col_70', 'col_71', 'col_72', 'col_73', 'col_75', 'col_76', 'col_77',

       'col_78', 'col_80', 'col_81', 'col_83', 'col_84', 'col_86']]



scaled_X_test = scalar.fit_transform(x)



prediction=rfc.predict(scaled_X_test)

data= { "id":df2["id"],"target":prediction}

#print(type(prediction))
submission = pd.DataFrame(data)

submission.to_csv('./submission.csv',index=None)

submission.describe()
