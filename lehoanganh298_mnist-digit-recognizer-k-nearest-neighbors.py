import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
import time
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# Only take 5000 first image because of train and predict time of KNN take too long
train_X=train_data[:5000].drop(['label'],axis=1).values
train_y=train_data[:5000]['label'].values
X_train,X_val,y_train,y_val = train_test_split(train_X,train_y,test_size=0.3,random_state=1)
# Choose n_neighbors = 5, weights=distance
start_time=time.time()
clf = neighbors.KNeighborsClassifier(n_neighbors=5,p=2,weights='distance')
clf.fit(X_train,y_train)
pred_val = clf.predict(test_data)
end_time=time.time()

print("Test set predict time: {}s.".format(end_time-start_time))

Label = pd.Series(pred_val,name = 'Label')
ImageId = pd.Series(range(1,28001),name = 'ImageId')
submission = pd.concat([ImageId,Label],axis = 1)
submission.to_csv('submission.csv',index = False)