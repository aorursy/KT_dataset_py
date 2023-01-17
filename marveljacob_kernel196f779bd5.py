# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
#print(os.listdir("../input"))
data=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
sample=pd.read_csv('../input/sample_submission.csv')

train,validate=train_test_split(data,test_size=0.3,random_state=100)
train_x = train.drop('label', axis=1)
train_y = train['label']
validate_x = validate.drop('label', axis=1)
validate_y = validate['label']
#checking every model and selecting which has better accuracy
model_dt=DecisionTreeClassifier()
model_dt.fit(train_x,train_y)
pred = model_dt.predict(validate_x)
print('Decision Tree accuracy: ',accuracy_score(validate_y,pred))
model_rf=RandomForestClassifier()
model_rf.fit(train_x,train_y)
pred1 = model_rf.predict(validate_x)
print('Random Forest accuracy: ',accuracy_score(validate_y,pred1))
model_ab=AdaBoostClassifier()
model_ab.fit(train_x,train_y)
pred2 = model_ab.predict(validate_x)
print('Adaboost accuracy: ',accuracy_score(validate_y,pred2))
pred_test=model_rf.predict(test)
df_dig_rec=pd.DataFrame(pred_test,columns=['label'])
df_dig_rec['ImageID']=test.index+1
df_dig_rec.head()
df_dig_rec[['ImageID','label']].to_csv('Submission.csv',index=False)

pixel=np.array(train.iloc[4].values[1:])
pixel=pixel.reshape(28,28)
import matplotlib.pyplot as plt
plt.imshow(pixel)