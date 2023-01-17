# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats



%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
column_names = ['id','col_0','col_1','col_2','col_3','col_4','col_5','col_6','col_7','col_8','col_9','col_10','col_11','col_12','col_13','col_14','col_15','col_16','col_17','col_18','col_19','col_20','col_21','col_22','col_23','col_24','col_25','col_26','col_27','col_28','col_29','col_30','col_31','col_32','col_33','col_34','col_35','col_36','col_37','col_38','col_39','col_40','col_41','col_42','col_43','col_44','col_45','col_46','col_47','col_48','col_49','col_50','col_51','col_52','col_53','col_54','col_55','col_56','col_57','col_58','col_59','col_60','col_61','col_62','col_63','col_64','col_65','col_66','col_67','col_68','col_69','col_70','col_71','col_72','col_73','col_74','col_75','col_76','col_77','col_78','col_79','col_80','col_81', 'col_82','col_83','col_84','col_85','col_86','col_87','target']

df = pd.read_csv("../input/minor-project-2020/train.csv",header=None, sep=',', delimiter=None, names=column_names)
len(df)
df=df.drop(0)

df = df.astype(np.float64) 
df.info()
y_train = df['target']

X_train = df.drop(["target"], axis=1)
#Feature Scaling for train set

from sklearn import preprocessing



scaler = preprocessing.MinMaxScaler()              #Instantiate the scaler

scaled_X_train = scaler.fit_transform(X_train)     #Fit and transform the data



scaled_X_train
from sklearn.linear_model import LogisticRegression



# all parameters not specified are set to their defaults

logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
column_names1 = ['id','col_0','col_1','col_2','col_3','col_4','col_5','col_6','col_7','col_8','col_9','col_10','col_11','col_12','col_13','col_14','col_15','col_16','col_17','col_18','col_19','col_20','col_21','col_22','col_23','col_24','col_25','col_26','col_27','col_28','col_29','col_30','col_31','col_32','col_33','col_34','col_35','col_36','col_37','col_38','col_39','col_40','col_41','col_42','col_43','col_44','col_45','col_46','col_47','col_48','col_49','col_50','col_51','col_52','col_53','col_54','col_55','col_56','col_57','col_58','col_59','col_60','col_61','col_62','col_63','col_64','col_65','col_66','col_67','col_68','col_69','col_70','col_71','col_72','col_73','col_74','col_75','col_76','col_77','col_78','col_79','col_80','col_81', 'col_82','col_83','col_84','col_85','col_86','col_87']



df1 = pd.read_csv("../input/minor-project-2020/test.csv",header=None, sep=',', delimiter=None, names=column_names1)
len(df1)
df1
df1=df1.drop(0)

len(df1)
df1
df1 = df1.astype(np.float64) 

df1.info()
df1 = df1.astype(np.int64) 

df1.info()
#Feature Scaling for test set

X_test = df1

from sklearn import preprocessing



scaler = preprocessing.MinMaxScaler()              #Instantiate the scaler

scaled_X_test = scaler.fit_transform(X_test)     #Fit and transform the data



scaled_X_test
predictions = logisticRegr.predict(df1)

print(predictions)

len(predictions)
df_pred = pd.DataFrame(data=predictions, columns=["target"])

df_pred
df_pred.index = np.arange(1, len(df_pred) + 1)

df_pred
test_id = df1['id']

len(test_id)

print(test_id)
df_pred
df_pred.insert(0, 'id',test_id , True)

df_pred
predictionLR = df_pred.to_csv('predictionLR.csv',index=False)
from sklearn.metrics import classification_report, confusion_matrix,plot_confusion_matrix