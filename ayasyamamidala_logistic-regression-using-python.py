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

from sklearn.linear_model import LinearRegression,LogisticRegression #used to import linear & logistic reg
df=pd.read_csv("../input/titanic/train_and_test2.csv")

df=df.dropna()# removes null values(if any)

df.shape #gives no.of rows & coloumns

df
df.keys() #it gives name of the coloumns 
x=df[['Passengerid', 'Age', 'Fare', 'Sex', 'sibsp']].values #independent variable

y=df[['2urvived']].values #dependent variable
lr_model=LinearRegression()

lr_model.fit(x,y)
y_pred=lr_model.predict(x)
exp=np.exp(-y_pred)+1

log=1/exp #1/1+e^(-y_pred)
y_con=df['2urvived']>0 #for all the values >0 it shows true.

y_con
df["tf"]=df['2urvived']>0

df.tf.value_counts() #gives no of true & false.
log_reg=LogisticRegression()

log_reg.fit(x,y_con) # machine is trained with logistic regression
log_reg.predict([[2,38.0,71.2833,1,1]]) #predicting the values for independent values
log_reg.predict_proba([[2,38.0,71.2833,1,1]])