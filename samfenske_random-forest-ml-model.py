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
random=np.random.rand(20)

df=pd.DataFrame(random,columns={'Number'})

df['Outcome']=df['Number']>0.7

df['Dummy']=np.random.rand(20)

df=df[['Number','Dummy','Outcome']]

from sklearn.model_selection import train_test_split

y=df['Outcome']

features=['Number','Dummy']

X=df[features]

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)
val_y
from sklearn.ensemble import RandomForestRegressor



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)



# fit your model



rf_model.fit(train_X,train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_predictions
rf_val_predictions.round()
val_y
df=val_X

df['Outcome']=val_y

df['Prediction']=rf_val_predictions

df['Predictions adjusted']=rf_val_predictions.round()

df['Correct']=df['Predictions adjusted']==df['Outcome']

df
def test_model(n,test_size):

    random=np.random.rand(n)

    df=pd.DataFrame(random,columns={'Number'})

    df['Outcome']=df['Number']>0.7

    df['Dummy']=np.random.rand(n)

    # df=df[['Number','Dummy','Outcome']]

    y=df['Outcome']

    features=['Number','Dummy']

    X=df[features]

    train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=test_size)

    rf_model = RandomForestRegressor(random_state=1)

    rf_model.fit(train_X,train_y)

    rf_val_predictions = rf_model.predict(val_X)

    df2=val_X

    df2['Outcome']=val_y

    df2['Prediction']=rf_val_predictions

    df2['Predictions adjusted']=rf_val_predictions.round()

    df2['Correct']=df2['Predictions adjusted']==df2['Outcome']

    return df2
test_model(30,.2)