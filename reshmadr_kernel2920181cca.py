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
df=pd.read_csv('../input/smarthomedataset/smart_home_dataset.csv',names=['Start_time','Location','Time_of_a_day','Object','Posture','Duration','Activity'],header=0)
df['timestamp']=pd.to_datetime(df.Start_time,format='%I:%M:%S %p')
df1=df[['Location','Time_of_a_day','Object','Posture','Activity']]
df1=pd.get_dummies(df1)
df1[['Start_time','Duration','timestamp']]=df[['Start_time','Duration','timestamp']]
X=df1[['Location_Bathroom', 'Location_Bedroom', 'Location_Front',

       'Location_Kitchen', 'Location_Toilet', 'Time_of_a_day_Afternoon',

       'Time_of_a_day_Evening', 'Time_of_a_day_Morning', 'Time_of_a_day_Night']]
y=df1[['Object_Cups cupboard', 'Object_Dishwasher', 'Object_Freezer',

       'Object_Fridge', 'Object_Frontdoor', 'Object_Groceries Cupboard',

       'Object_Hall-Bathroom door', 'Object_Hall-Bedroom door',

       'Object_Hall-Toilet door', 'Object_Pans Cupboard',

       'Object_Plates cupboard', 'Object_ToiletFlush',]]
from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
linear = LogisticRegression()
model= DecisionTreeClassifier()
model.fit(Xtrain,ytrain)
y1=model.predict(Xtest)

output=pd.DataFrame(y1,columns=['Object_Cups cupboard', 'Object_Dishwasher', 'Object_Freezer',

       'Object_Fridge', 'Object_Frontdoor', 'Object_Groceries Cupboard',

       'Object_Hall-Bathroom door', 'Object_Hall-Bedroom door',

       'Object_Hall-Toilet door', 'Object_Pans Cupboard',

       'Object_Plates cupboard', 'Object_ToiletFlush',])
output
model.score(X,y)