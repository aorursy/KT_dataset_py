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
df=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df_raw =df.copy()
df.head(10)
df.shape
df.dtypes
df['salary'] = df['salary'].fillna(0)
df.head(10)
df.isnull().sum()
df.dtypes
df['status']=label_encoder.fit_transform(df['status'])

from sklearn.model_selection import train_test_split

x= df[df.loc[:,df.columns != 'status'].columns]
y=df['status']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
# Get list of categorical variables
s = (x_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)
from sklearn.preprocessing import LabelEncoder


#make copy
#label_x_train=x_train.copy()
#label_x_test=x_test.copy()


label_encoder=LabelEncoder()

for col in object_cols:
    x_train[col]=label_encoder.fit_transform(x_train[col])
    x_test[col]=label_encoder.transform(x_test[col])
    
#encoded variables:

#gender M=1
#ssc_b central=0
#hsc_b central=0
#hsc_stream science=2  arts=0  comm=1
#degree_t comm&mngm=0 scie&tech=2 arts=1
#workex yes=1  no=0
#speci mark&fin=0 mark&hr=1
x_train.head(20)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(x_train, x_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=50, random_state=0)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    return mean_absolute_error(y_test, preds)


print("MAE from  (Label Encoding):") 
print(score_dataset(x_train, x_test, y_train, y_test))

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(x_train,y_train)
pred =model.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred) 
output =pd.DataFrame({'ID': x_test.index,
                     'status': pred})
output.to_csv('submission.csv',index=False)
#not placed =0,placed=1
out =pd.read_csv('./submission.csv')
out.head(20)