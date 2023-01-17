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
#Import Libs

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
# Reading in the dataset

df = pd.read_csv("../input/fetal-health-classification/fetal_health.csv")

df.head()
#Counting labels

df["fetal_health"].value_counts()
##Missing values

df.isnull().sum()
df.dtypes
column_names = [c for c in df.columns]
print(column_names)
for c in column_names:

    try:

        plt.title("Distribution of {}".format(c))

        sns.distplot(df[c])

        plt.show()

    except:

        print("Not Eligible")
df.shape
for c in column_names:

    print(c)

    print(df[c].nunique())
cat_columns = [c for c in column_names if c != "fetal_health" and df[c].nunique() < 9 ]
cat_columns
for c in cat_columns:

    c_dum = pd.get_dummies(df[c],drop_first=True)

    df = pd.concat([df,c_dum],axis=1)

    df.drop([c],inplace=True,axis=1)
df.head()
df.shape
from sklearn.model_selection import train_test_split

X = df.drop(["fetal_health"],axis=1).values

y = df["fetal_health"].values
X_train, X_test , y_train , y_test = train_test_split(X,y,test_size=0.30,random_state=42)
X_train.shape, X_test.shape
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_X_train = scaler.fit_transform(X_train)

scaled_X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(scaled_X_train,y_train)
log_pred = log_model.predict(scaled_X_test)
from sklearn import metrics
print(metrics.classification_report(y_test,log_pred))
## svm
from sklearn.svm import SVC
sup_vec = SVC()
sup_vec.fit(scaled_X_train,y_train)
svc_pred = sup_vec.predict(scaled_X_test)
print(metrics.classification_report(y_test,svc_pred))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(scaled_X_train,y_train)
rfc_preds = rfc.predict(scaled_X_test)
print(metrics.classification_report(rfc_preds,y_test))
from xgboost import XGBClassifier
XGB = XGBClassifier()

XGB.fit(scaled_X_train,y_train)

XGB_preds = XGB.predict(scaled_X_test)

print(metrics.classification_report(XGB_preds,y_test))
XGB.feature_importances_.shape
features = [c for c in column_names if c != "fetal_health"]
feat_df = pd.DataFrame(features,columns=["label"])

fat_imp_df = pd.DataFrame(XGB.feature_importances_,columns=["importance"])

feat_imp_df=feat_df.join(fat_imp_df)
feat_imp_df
plt.figure(figsize=(15,8))

sns.barplot(y="label",x="importance",order=feat_imp_df.sort_values("importance").label,data=feat_imp_df)