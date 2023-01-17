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
df=pd.read_csv("/kaggle/input/email-spam-classification-dataset-csv/emails.csv")
df
# seprating features and the target variables
x=df.iloc[:,1:3000]
y=df.iloc[:,3001]
# deviding the data into train and test data
from sklearn.model_selection import train_test_split as tts
train_x,test_x,train_y,test_y=tts(x,y,random_state=42,test_size=0.2)
# using multinomial navies bias
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(train_x,train_y)
model.score(train_x,train_y)
model.score(test_x,test_y)
y_pred=model.predict(test_x)
test_=f1_score(test_y,y_pred)
test_
