import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential

from keras.layers import Dense,Dropout

from sklearn.preprocessing import LabelBinarizer

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score



train_f = pd.read_csv("../input/kaggle/train.csv")

test_f = pd.read_csv("../input/kaggle/test.csv")



print ('Training data features:')

print (train_f.columns)

print ('\nTest data features:')

print (test_f.columns)



print (train_f.shape,'\n')

print (train_f.head(),'\n')

print (train_f.tail(),'\n')

print (train_f.describe(),'\n')



print (test_f.shape,'\n')

print (test_f.head(),'\n')

print (test_f.tail(),'\n')

print (test_f.describe(),'\n')



train_sc=train_f.values[:,0]

test_sc=test_f.values[:,0]



yy= train_f['type']

xx = pd.get_dummies(train_f.drop(['color','type','id'], axis = 1))



X_training, X_testing, y_training, y_testing = train_test_split(xx, yy, test_size = 0.2, random_state = 0)



XX=np.array(X_training)

yy=np.array(y_training)



lovy=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,), random_state=0).fit(XX,yy)

print(lovy)

y_pred=lovy.predict(X_testing)



y_pred=lovy.predict(X_testing)

lovy.score(X_testing,y_testing)



total= cross_val_score(lovy, XX, yy, cv=5)

total

total.mean()



testing=test_f.drop(['color','id'],axis=1)

prdt=lovy.predict(testing)



final=pd.DataFrame({'id':test_sc, 'type': prdt})

final.to_csv("lovish_final.csv",index=False)