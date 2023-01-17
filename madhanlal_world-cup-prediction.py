# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("../input/ODI_Data_Final1.csv")
df.head()
df.shape
df.info()
df.isnull().sum()
df_X = df.loc[:,['Country','Opposition','Ground','Toss','Inns','Score','Wickets','Overs','Target']]
df_X.Target.fillna(999,inplace=True)
df_y = df['Result']
X = pd.get_dummies(df_X, drop_first=True)
from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(df_y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

#Testing Model

print("Logistic Regression Accuracy Score =", log_reg.score(X_test, y_test)*100)
test_df = pd.DataFrame([["England","New Zealand","Lord\'s","lost",2,170,4,40,243]],

             columns= ['Country','Opposition','Ground','Toss',

                       'Inns','Score','Wickets','Overs','Target'])

test_df
df_X1=pd.concat([df_X,test_df], axis=0)
X1 = pd.get_dummies(df_X1,drop_first=True)
X11 = X1.iloc[-1:,]
y_predict = log_reg.predict(X11)
y_predict = pd.DataFrame(y_predict, columns=["Result"]).replace({1:"win",0:"loss"})

y_predict
y_prob = (log_reg.predict_proba(X11))*100

y_prob = pd.DataFrame(y_prob,columns=["loss%","win%"])

y_prob
y_prob = (log_reg.predict_proba(X11))*100

y_prob = pd.DataFrame(y_prob,columns=["loss%","win%"])

y_prob
pd.concat([test_df,y_predict,y_prob],axis=1)
test_df = pd.DataFrame([["India","New Zealand","Manchester","won",1,250,6,50,0.0]],

             columns= ['Country','Opposition','Ground','Toss',

                       'Inns','Score','Wickets','Overs','Target'])

test_df
df_X1=pd.concat([df_X,test_df], axis=0)
X1 = pd.get_dummies(df_X1,drop_first=True)

X11 = X1.iloc[-1:,]
y_predict = log_reg.predict(X11)
y_predict = pd.DataFrame(y_predict, columns=["Result"]).replace({1:"win",0:"loss"})

y_predict
y_prob = (log_reg.predict_proba(X11))*100

y_prob = pd.DataFrame(y_prob,columns=["loss%","win%"])

y_prob
pd.concat([test_df,y_predict,y_prob],axis=1)