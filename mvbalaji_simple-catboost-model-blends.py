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
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score
#Upload data

df= pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/train.csv")

df["train_or_test"]="train"

test = pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/test.csv")

test["train_or_test"]="test"
df=df.append(test)
df['Gender'].value_counts()
df['Vehicle_Age'].value_counts()
df['Vehicle_Damage'].value_counts()
#converting object to int type

df['Vehicle_Age']=df['Vehicle_Age'].replace({'< 1 Year':0,'1-2 Year':1,'> 2 Years':2})

df['Gender']=df['Gender'].replace({'Male':1,'Female':0})

df['Vehicle_Damage']=df['Vehicle_Damage'].replace({'Yes':1,'No':0})
df.info()
df.describe()
#binning of annual premium

df["Annual_bin"]=pd.cut(df["Annual_Premium"],bins=np.arange(0,550000,1000),labels=np.arange(1,550))
df.isna().sum()
col_1=['Gender', 

       'Age', 

       'Driving_License', 

       'Region_Code', 

       'Previously_Insured', 

       'Vehicle_Age', 

       'Vehicle_Damage', 

       'Annual_Premium', 

       'Policy_Sales_Channel',

       "Annual_bin"]



# categorical columns

cat_col=['Gender',

         'Driving_License', 

         'Region_Code', 

         'Previously_Insured', 

         'Vehicle_Damage',

         'Policy_Sales_Channel',

         "Annual_bin"] 
df['Region_Code']=df['Region_Code'].astype(int)

df['Policy_Sales_Channel']=df['Policy_Sales_Channel'].astype(int)

df['Annual_bin']=df['Annual_bin'].astype(int)
X=df[df.train_or_test=="train"].drop(["id","train_or_test","Vintage","Response"],axis=1)

y=df[df.train_or_test=="train"]['Response']
X_test=df[df.train_or_test=="test"].drop(["id","train_or_test","Vintage","Response"],axis=1)
X_test.head(5)
cat_pred=pd.DataFrame()



for rand in [1111111, 22222222,

             1150303, 46584658, 202020, 1919191919,

             90909090,91919191,92929292,12121212,21212121,

             123456789,22222222,33333333,44444444,55555555,

             987654321, 17923798, 32763271, 34263748, 89674523,

             123454321,223456543,33445566,44556677,55667788,

             1990,1991, 2000, 2020, 2021,

             2031,2032,2033,2034,2035]:

    

    X_t, X_tt, y_t, y_tt = train_test_split(X, y, test_size=.25, random_state=rand,

                                            stratify=y,shuffle=True)

    

    catb = CatBoostClassifier(n_estimators=10000,

                       random_state=rand,

                       eval_metric='Accuracy',

                       learning_rate=0.08,

                       depth=8,

                       bagging_temperature=0.3,

                       task_type='GPU'

                       )

    

    catb=catb.fit(X_t, y_t,cat_features=cat_col,eval_set=(X_tt, y_tt),early_stopping_rounds=30,verbose=100)

    col_name="CATB_"+str(rand)

    cat_pred[col_name]=catb.predict_proba(X_test)[:, 1]
cat_pred["Response"]=cat_pred[cat_pred.columns].mean(axis=1)

cat_pred["id"]=test["id"]

cat_pred[["id","Response"]].to_csv("submission_blend.csv", index=False)