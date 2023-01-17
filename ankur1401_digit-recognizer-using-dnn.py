import numpy as np

import pandas as pd
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
X=train.iloc[:,1:].values
y=train.iloc[:,0].values
from sklearn.model_selection import GridSearchCV,train_test_split

from sklearn.neural_network import MLPClassifier
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.33,random_state=62)
#Here we have used 512 neurons for our hidden layer.

mlp=MLPClassifier([512],activation='relu',solver='adam',learning_rate_init=0.001,)
mlp.fit(X_train,y_train)
mlp.score(X_train,y_train)
mlp.score(X_val,y_val)
mlp.fit(X,y)
pred=mlp.predict(test.values).reshape(-1,1)
output=np.concatenate((np.arange(1,test.shape[0]+1).reshape(-1,1),pred),axis=1)
submission=pd.DataFrame(output,columns=['ImageId','Label'])
submission.to_csv('Submission.csv',index=False)    #Generating submission file.