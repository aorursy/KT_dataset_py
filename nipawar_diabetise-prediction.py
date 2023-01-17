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
df=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head(20)
df.dtypes
df.shape
df.isnull().sum()
from sklearn.model_selection import train_test_split



x=df[df.loc[:,df.columns != 'Outcome'].columns]

y=df['Outcome']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100,random_state=0)
from sklearn.metrics import mean_absolute_error



def score_model(model, x_t=x_train, x_v=x_test, y_t=y_train, y_v=y_test):

    model.fit(x_t,y_t)

    pred = model.predict(x_v)

    return mean_absolute_error(y_v,pred)



mae = score_model(model)

print("model mae:",mae)
model.fit(x_train,y_train)

pred_test=model.predict(x_test)





output=pd.DataFrame({'ID':x_test.index,

                    'output':pred_test})



output.to_csv('dia_submission.csv',index=False)
out_df=pd.read_csv('./dia_submission.csv')
out_df.head(10)
out_df.round()