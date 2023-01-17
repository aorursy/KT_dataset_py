import pandas as pd
import numpy as np
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
df_test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
df_val = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')
x_train = df_train.iloc[:,1:]/255.
y_train = df_train['label']

x_test = df_test.iloc[:,1:]/255.

x_val = df_val.iloc[:,1:]/255.
y_val = df_val['label']
class knn():
    def __init__(self,k = 5):
        self.k = k
        self.x = []
        self.y = []
        
    def fit(self,train_x,label):
        
        self.x = np.array(train_x)
        self.y = np.array(label)
    
    def predict(self, test_x):

        test_x = np.array(test_x)
        pred_y = []
        
        for x in test_x:
            counter = np.zeros(10)

            diff_x = self.x - x
            dist = np.sum(diff_x ** 2,axis = 1)
            sort_dist = np.argsort(dist)
            
            for k in range(self.k):
                counter[self.y[sort_dist[k]]] += 1
            pred_y.append(np.argmax(counter))
            
        return pred_y
    
    def scores(self,test_x,label):
        
        pred_y = self.predict(test_x)
        return np.sum(pred_y == label)/len(pred_y)  
from time import process_time_ns 
cls = knn(5)
cls.fit(x_train,y_train)

t1_start = process_time_ns()

pred_val = cls.predict(x_val)
print(cls.predict(x_val))

t1_end = process_time_ns()
print("Elaspe time: ",t1_end - t1_start)
print('Validation score: ',np.sum(y_val == pred_val)/len(y_val))
t2_start = process_time_ns()

pred_test = cls.predict(x_val)

t2_end = process_time_ns()
print("Test Elaspe time:%f",t1_end - t1_start)
pd_outcome = pd.DataFrame({'id':[i for i in range(len(pred_test))],'label':pred_test})
pd_outcome.to_csv('test_submission_knn.csv',index = False)
