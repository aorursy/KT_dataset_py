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
'''THE INTRODUCTION THAT HOW MODEL WORKS

**FOR EXAMpLE:my best friend has a real estate buisiness and he earns lot of money but he had a challenge that as his buisiness is increasing

he had difficulty in predicting the cost of the houses because each house consists of many variations...so he consulted

me as i am a data scientist he asked me predict the houses

**so the matter comess here as my friend told the intuition of previous data with that i can predict the unseen data

**in this problem statement i will be going to use desicion tree as thre are many other models but for now i am going to use DT...

because decision tree is easy to understand..

**for now we are creating a decision tree which consists of two splits  like if we have many variables like lot size,car parking

,arae,etc...then we can create many more splits...'''
import pandas as pd

train_and_test2 = pd.read_csv("../input/titanic/train_and_test2.csv",index_col='Passengerid')
full_data = train_and_test2
full_data.describe()
full_data.columns
survived = full_data['2urvived'] 
full_data['Embarked'].isnull()

full_data = full_data.dropna(axis=0)
full_data.rename(columns={'2urvived':'survived',}, 

                 inplace=True)
full_data.columns
y = full_data.survived
full_data.head(10)
full_data.Fare
features = ['Age', 'Fare','Sex', 'sibsp','Pclass','Embarked']
x = full_data[features]
x.describe()
x.head()
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=1)
model.fit(x,y)
print('predicting the first five rows:')

print(x.head(5))

print('the predictions are:')

print(model.predict(x.head(5)))

from sklearn.metrics import mean_absolute_error
predictions = model.predict(x)
mean_absolute_error(y,predictions)
from sklearn.model_selection import train_test_split
train_x,val_x,train_y,val_y = train_test_split(x,y,random_state = 1)
my_model = model.fit(train_x,train_y)
my_model
predict = my_model.predict(val_x)
mean_absolute_error(val_y,predict)
def get_mae(max_leaf_nodes,train_x,val_x,train_y,val_y):

    my_model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)

    my_model.fit(train_x,train_y)

    preds = my_model.predict(val_x)

    mae = mean_absolute_error(val_y,preds)

    return mae

    
for max_leaf_nodes in [5,25,50,100]:

    my_mae = get_mae(max_leaf_nodes,train_x,val_x,train_y,val_y)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
get_mae(500,train_x,val_x,train_y,val_y)
for max_leaf_nodes in [200,300,4000]:

    my_mae = get_mae(max_leaf_nodes,train_x,val_x,train_y,val_y)

    print("max leaf nodes: %d \t\t mean absolute error: %d"%(max_leaf_nodes,my_mae))
from sklearn.ensemble import RandomForestClassifier
my_model1 = RandomForestClassifier(random_state=1)
my_model1.fit(train_x,train_y)
pred = my_model1.predict(val_x)

mean_absolute_error(val_y,pred)