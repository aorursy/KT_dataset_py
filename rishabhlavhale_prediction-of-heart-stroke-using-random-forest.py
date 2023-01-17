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
df=pd.read_csv('/kaggle/input/health-care-data-set-on-heart-attack-possibility/heart.csv')
df.head()
df.info()
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier
df.head()
predictors = df.drop(['target'], axis=1)
predictors.head()
target = df["target"] 
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)

filename = 'model.sav'        #trained model save in this file
pickle.dump(randomforest, open(filename, 'wb'))
def prediction_model(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal):
    
    x= [[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]
    randomforest=pickle.load(open('model.sav','rb'))
    predictions =randomforest.predict(x)
    print(predictions)
prediction_model(50,0,2,120,219,0,1,158,0,1.6,1,0,2)
prediction_model(61,1,0,138,166,0,0,125,1,3.6,1,1,2)
