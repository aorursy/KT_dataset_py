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
data=pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',delimiter=',',encoding='latin-1')
data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
data.head()
text=data['v2']
print(text[0],text[1])
from sklearn.feature_extraction.text import TfidfVectorizer
c=TfidfVectorizer()
text=c.fit_transform(text).toarray()
print(text[0])
print(text[1])
df=pd.DataFrame(text)
df['Output']=data['v1'].replace({'ham':0,'spam':1})
df.head()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

threshold=-1.5
y=df.pop('Output')
x=df
x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True)
model=LinearRegression(fit_intercept=True,normalize=True,n_jobs=-1)
model.fit(x_train,y_train)
net_accu=0
net_threshold=0

while(threshold<=5.5):
    y_pre=model.predict(x_test)
    y_pre[y_pre>threshold]=1
    y_pre[y_pre<=threshold]=0
    tmp_accu=accuracy_score(y_pre,y_test)
    print('Threshold: ',threshold,' Accuracy: ',tmp_accu)
    if(tmp_accu>net_accu):
        net_accu=tmp_accu
        net_threshold=threshold
    threshold+=0.01

print('\nReqd Threshold: ',net_threshold,' Reqd Accuracy: ',net_accu)
