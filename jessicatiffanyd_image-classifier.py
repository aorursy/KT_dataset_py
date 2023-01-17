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
#import dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
%matplotlib inline
#using pandas to read the database stored in the same folder
data=pd.read_csv('../input/train.csv')
data.head()
#extracting data from the dataset and viewing them u close
a=data.iloc[3,1:].values
#reshaping the  extracted data
a=a.reshape(28,28).astype('uint8')
plt.imshow(a)
#preparing data
#seperating lables and data values
df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]
#creating test and train sizes
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
#check data
y_train.head()
#call rf classifier
rf=RandomForestClassifier(n_estimators=100)
#fit the model
rf.fit(x_train, y_train)
#predictions on test data
pred=rf.predict(x_test)
pred
#check prediction accuracy
s=y_test.values

#calculte the no. of corresctly predicted values
count=0
for i in range (len(pred)):
    if pred[i] == s[i]:
        count=count+1
count
#total values the prediction code was run on
len(pred)
#accuracy value
8082/8400