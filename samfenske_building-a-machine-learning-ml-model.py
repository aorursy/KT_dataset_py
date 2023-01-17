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
#array of 10 random numbers between 0 and 1
random=np.random.rand(5)
random
#add array to a dataframe
df=pd.DataFrame(random,columns={'Number'})
df
#so pandas stops showing an unnecessary error
pd.options.mode.chained_assignment = None
#add true/false column reflecting whether or not the number is greater than 0.7
df['Outcome']=df['Number']>0.7
df
#add another column of random numbers that has no effect on the outcome, call this 'dummy data'
df['Dummy']=np.random.rand(5)
df
#manually change order of columns, looks better if 'Number' and 'Dummy' columns come before the 'Outcome' column
df=df[['Number','Dummy','Outcome']]
df
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

#designate an attribute to predict/measure
y=df['Outcome']
y
#designate which elements you want to influence the prediction
features=['Number','Dummy']
X=df[features]
X
#split into training and validation data
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)
train_X
train_X
train_y
val_X
val_y
#create model
basic_model = DecisionTreeRegressor(random_state=1)

#fit the model with the training data
basic_model.fit(train_X, train_y)

#make predictions based on validation data
val_predictions = basic_model.predict(val_X)
val_predictions
# add the predictions to the dataframe of validation data
val_Xdf=pd.DataFrame(val_X)
val_Xdf['Outcome']=val_y
val_Xdf['Prediction']=val_predictions
val_Xdf['Correct?']=val_Xdf['Outcome']==val_Xdf['Prediction']
val_Xdf
1.0==True
0.0==False
#make dataframe composed of only data in which the prediction was correct
#the length of this dataframe will thus be the number of correct predictions
num_true=len(val_Xdf[val_Xdf['Correct?'].isin([True])])

#the total number of predictions will be equal to the length of the full dataframe
total=len(val_Xdf)

#divide number of true predictions by the total number of predictions to get ratio
correct_ratio=num_true/total
correct_ratio
def ratio(df):
    num_true=len(df[df['Correct?'].isin([True])])
    total=len(df)
    correct_ratio=num_true/total
    print(str(num_true)+'/'+str(total)+' or '+str(100*correct_ratio)+'%')
ratio(val_Xdf)
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
#     #num_true=len(df[df['Correct?'].isin([True])])
    #total=len(df)
    #correct_ratio=num_true/total
    #return correct_ratio
    return val_Xdf
model_test(70,0.1)
ratio(model_test(70,0.1))
ratio(model_test(70,0.4))