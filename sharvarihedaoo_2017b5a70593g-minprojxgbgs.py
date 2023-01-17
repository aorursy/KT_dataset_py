#This is a trial, I could not submit these results in time because it tooktoo long to run
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
#Generating train dataframe
column_names = ['id','col_0','col_1','col_2','col_3','col_4','col_5','col_6','col_7','col_8','col_9','col_10','col_11','col_12','col_13','col_14','col_15','col_16','col_17','col_18','col_19','col_20','col_21','col_22','col_23','col_24','col_25','col_26','col_27','col_28','col_29','col_30','col_31','col_32','col_33','col_34','col_35','col_36','col_37','col_38','col_39','col_40','col_41','col_42','col_43','col_44','col_45','col_46','col_47','col_48','col_49','col_50','col_51','col_52','col_53','col_54','col_55','col_56','col_57','col_58','col_59','col_60','col_61','col_62','col_63','col_64','col_65','col_66','col_67','col_68','col_69','col_70','col_71','col_72','col_73','col_74','col_75','col_76','col_77','col_78','col_79','col_80','col_81', 'col_82','col_83','col_84','col_85','col_86','col_87','target']
df = pd.read_csv("../input/minor-project-2020/train.csv",header=None, sep=',', delimiter=None, names=column_names)
#dropping 0th row 
df=df.drop(0)
df
#Data preprocessing
df = df.astype(np.float64)
df = df.astype(np.int64)
df.info()
y_train = df['target']
X_train = df.drop(["target"], axis=1)
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
scaled_X_train = scalar.fit_transform(X_train)

scaled_X_train
#Using xg_boost XGBClassifier

from xgboost import XGBClassifier
xgb = XGBClassifier()

xgb.fit(scaled_X_train, y_train)
#generating test dataframe

column_names1 = ['id','col_0','col_1','col_2','col_3','col_4','col_5','col_6','col_7','col_8','col_9','col_10','col_11','col_12','col_13','col_14','col_15','col_16','col_17','col_18','col_19','col_20','col_21','col_22','col_23','col_24','col_25','col_26','col_27','col_28','col_29','col_30','col_31','col_32','col_33','col_34','col_35','col_36','col_37','col_38','col_39','col_40','col_41','col_42','col_43','col_44','col_45','col_46','col_47','col_48','col_49','col_50','col_51','col_52','col_53','col_54','col_55','col_56','col_57','col_58','col_59','col_60','col_61','col_62','col_63','col_64','col_65','col_66','col_67','col_68','col_69','col_70','col_71','col_72','col_73','col_74','col_75','col_76','col_77','col_78','col_79','col_80','col_81', 'col_82','col_83','col_84','col_85','col_86','col_87']

df1 = pd.read_csv("../input/minor-project-2020/test.csv",header=None, sep=',', delimiter=None, names=column_names1)
df1=df1.drop(0)
df1
#Data preprocessing
df1 = df.astype(np.float64)
df1 = df.astype(np.int64)
df1.info()
X_test = df1
X_test
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
scaled_X_test = scalar.fit_transform(X_test)

scaled_X_test
#Using xg_boost XGBClassifier

from xgboost.sklearn import XGBClassifier
#from sklearn import cross_validation, metrics
#from sklearn.grid_search import GridSearchCV
xgb = XGBClassifier()

xgb1=XGBClassifier(col_sample_by_tree=0.8,learning_rate =0.1,max_delta_step=4,max_depth=4,min_child_weight=0.8,n_estimators=500,objective='binary:logistic',scale_pos_weight=1,seed=27)
xgb.fit(X_train,y_train)
param_test1 = {
 'max_depth':[3,5,10],'min_child_weight':[3,5,10],'max_delta_step' : [1,4,8],'objective' : ['binary:logistic']}  

from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(xgb, param_test1, verbose=1,n_jobs=-1)
clf.fit(scaled_X_train,y_train)
y_pred1=xgb.predict(scaled_Xtest)
predictions = pd.DataFrame(data=y_pred1, columns=["target"])
predictions
predictions.index = np.arange(1, len(predictions)+1)
predictions
test_id = df1['id']
predictions.insert(0, 'id',test_id , True) 
predictions
predictionXG = predictions.to_csv('predictionXG.csv',index=False)