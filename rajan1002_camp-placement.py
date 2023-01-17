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
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,confusion_matrix

import seaborn as sns

import statsmodels.api as sm

from statsmodels.formula.api import ols

import warnings

warnings.filterwarnings("ignore")

df1 = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

df=df1.copy(deep=True)

print(df)
df.isnull().sum()
df["salary"].describe()
m=round(df["salary"].mean(),0)

df["salary"].replace(np.nan,m,inplace=True)

df.drop("sl_no",axis=1,inplace=True)
df["gender"].value_counts()

dic={"M":0,"F":1}

df["gender"]=df["gender"].map(dic)
df["ssc_b"].value_counts()

dic={"Others":0,"Central":1}

df["ssc_b"]=df["ssc_b"].map(dic)
df["hsc_b"].value_counts()

dic={"Others":0,"Central":1}

df["hsc_b"]=df["hsc_b"].map(dic)

df["workex"].value_counts()

dic={"No":0,"Yes":1}

df["workex"]=df["workex"].map(dic)

df["specialisation"].value_counts()

dic={"Mkt&Fin":0,"Mkt&HR":1}

df["specialisation"]=df["specialisation"].map(dic)

df["status"].value_counts()

dic={"Placed":1,"Not Placed":0}

df["status"]=df["status"].map(dic)

df["degree_t"]=df["degree_t"].astype("object")

df=pd.get_dummies(df,drop_first=True)

df["s_d"]=df["ssc_p"]*df["degree_p"]

df["h_d"]=df["hsc_p"]*df["degree_p"]
model1=ols("mba_p~ssc_p+degree_p+s_d",data=df).fit()

print(model1.summary())
anova_table=sm.stats.anova_lm(model1,type=1)

print(anova_table)
model2=ols("mba_p~hsc_p+h_d",data=df).fit()

print(model2.summary())
anova_table=sm.stats.anova_lm(model2,type=1)

print(anova_table)
model3=ols("mba_p~ssc_p+degree_p+s_d+h_d",data=df).fit()

print(model3.summary())
anova_table=sm.stats.anova_lm(model3,type=1)

print(anova_table)
col=["status","salary","degree_t_Others","ssc_b","hsc_b"]



data=df.drop(col,axis=1)

x=data.values

y=df["status"].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1001)

lgr=LogisticRegression()

model_fit=lgr.fit(x_train,y_train)

prediction=lgr.predict(x_test)

print(accuracy_score(y_test,prediction))

print(confusion_matrix(y_test,prediction))