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
df_test = pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/test.csv")

df_train = pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/train.csv")

df_sub = pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/sample_submission.csv")
import matplotlib.pyplot as plt 
df_train.head()
y = df_train["Response"].copy()

df_train.drop(["Response"],axis=1,inplace=True)
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(df_train,y,random_state = 0)

cat_col = x_train.select_dtypes(include="object").columns

num_col = x_train.select_dtypes(exclude="object").columns[1:]

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer
preprocessing = ColumnTransformer(transformers=[("num_col",SimpleImputer(),num_col),("cat_col",OneHotEncoder(),cat_col)])
fit_trans = preprocessing.fit_transform(x_train.drop(["id"],axis=1))

print(fit_trans[0])

print(fit_trans[1])

x_train.drop(["id"],axis=1).head()
fig1,axes = plt.subplots(ncols=2, nrows=2,figsize=(16,12))

fig1.suptitle("Categorical Data")

axes[0,0].bar(df_train["Vehicle_Age"].value_counts().index,df_train["Vehicle_Age"].value_counts().values , width=0.5)

axes[0,0].set_xlabel("Vehicle Age")

axes[0,1].bar(df_train["Vehicle_Damage"].value_counts().index,df_train["Vehicle_Damage"].value_counts().values, width=0.5)

axes[0,1].set_xlabel("Vehicle Damage")

axes[1,0].bar(df_train["Previously_Insured"].value_counts().index,df_train["Previously_Insured"].value_counts(), width=0.5)

axes[1,0].set_xticks(df_train["Previously_Insured"].value_counts().index)

axes[1,0].set_xlabel("Previously Insured")

axes[1,1].bar(df_train["Gender"].value_counts().index,df_train["Gender"].value_counts().values, width=0.5)

axes[1,1].set_xlabel("Gender")



plt.show()
df_train.drop(["id"],axis=1,inplace=True)

df_test.drop(["id"],axis=1,inplace=True)
x_train.head()
enc = OneHotEncoder(handle_unknown='ignore',sparse=False)

enc_xtrain = pd.DataFrame(enc.fit_transform(x_train[cat_col]))

enc_xtest = pd.DataFrame(enc.transform(x_test[cat_col]))

enc_test = pd.DataFrame(enc.transform(df_test[cat_col]))
enc_xtrain.index = x_train.index

enc_xtest.index = x_test.index

enc_test.index = df_test.index

x_train.drop(cat_col,axis=1,inplace=True)

x_test.drop(cat_col,axis=1,inplace=True)

df_test.drop(cat_col,axis=1,inplace=True)

post_x_train = pd.concat([x_train,enc_xtrain],axis=1)

post_x_test = pd.concat([x_test,enc_xtest],axis=1)

post_df_test = pd.concat([df_test,enc_test],axis=1)

post_x_train.drop(["id"],axis=1,inplace=True)

post_x_test.drop(["id"],axis=1,inplace=True)
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
model = XGBClassifier(n_estimators=1000, learning_rate=0.5)
model.fit(post_x_train,y_train,early_stopping_rounds=20, 

             eval_set=[(post_x_test, y_test)], 

             verbose=False)
accuracy_score(model.predict(post_x_test),y_test)

#0.8775058250593002
model.predict(post_x_test).size

#95278
df_sub["Response"] = model.predict(post_df_test)
df_sub.to_csv('submission.csv', index=False)