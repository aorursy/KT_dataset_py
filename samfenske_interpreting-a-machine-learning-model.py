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
random=np.random.rand(5)

df=pd.DataFrame(random,columns={'Number'})

df['Outcome']=df['Number']>0.7

df['Dummy']=np.random.rand(5)

df=df[['Number','Dummy','Outcome']]

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

y=df['Outcome']

features=['Number','Dummy']

X=df[features]

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)

basic_model = DecisionTreeRegressor(random_state=1)

basic_model.fit(train_X, train_y)

val_predictions = basic_model.predict(val_X)

train_predictions = basic_model.predict(train_X)
def model_test(num_elements,test_size):

    random=np.random.rand(num_elements)

    df=pd.DataFrame(random,columns={'Number'})

    df['Outcome']=df['Number']>0.7

    df['Dummy']=np.random.rand(num_elements)

    y=df['Outcome']

    features=['Number','Dummy']

    X=df[features]

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1,test_size=test_size)

    final_model = DecisionTreeRegressor(random_state=0)

    final_model.fit(train_X, train_y)

    predictions=final_model.predict(val_X)

    val_Xdf=val_X

    val_Xdf['Outcome']=val_y

    val_Xdf['Prediction']=predictions

    val_Xdf['Correct?']=val_Xdf['Outcome']==val_Xdf['Prediction']

    return val_Xdf
model_test(30,.2)
def model_train(num_elements,test_size):

    random=np.random.rand(num_elements)

    df=pd.DataFrame(random,columns={'Number'})

    df['Outcome']=df['Number']>0.7

    df['Dummy']=np.random.rand(num_elements)

    y=df['Outcome']

    features=['Number','Dummy']

    X=df[features]

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1,test_size=test_size)

    final_model = DecisionTreeRegressor(random_state=0)

    final_model.fit(train_X, train_y)

    predictions=final_model.predict(train_X)

    train_Xdf=train_X

    train_Xdf['Outcome']=train_y

    train_Xdf['Prediction']=predictions

    train_Xdf['Correct?']=train_Xdf['Outcome']==train_Xdf['Prediction']

    return train_Xdf
model_train(30,.2)
from IPython.display import Image

Image('/kaggle/input/decision_tree.png')
df=pd.DataFrame(np.random.rand(50),columns={'Number'})

df['Outcome']=df['Number']>0.7

df['Dummy']=np.random.rand(50)

df=df[['Number','Dummy','Outcome']]

y=df['Outcome']

features=['Number','Dummy']

X=df[features]

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)
#machine learning algorithms operate by splitting data into trees (like stacking cups)

#max_leaf_nodes refers to the number of nodes/elements stored at the base of the tree

    #low values indicate very vague criteria for predictions 

    #high values indicate very specific criteria for predictions

final_model = DecisionTreeRegressor(random_state=1,max_leaf_nodes=8)



#train the model

final_model.fit(train_X, train_y)



#make predictions for the testing data

predictions=final_model.predict(val_X)



#check if predictions are correct

predictions==val_y
final_model = DecisionTreeRegressor(random_state=1,max_leaf_nodes=2)

final_model.fit(train_X, train_y)

predictions=final_model.predict(val_X)

predictions==val_y