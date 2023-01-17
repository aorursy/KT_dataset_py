import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

import os
print(os.listdir("../input/data-science-london-scikit-learn"))

train_data = pd.read_csv('../input/data-science-london-scikit-learn/train.csv',header = None)
train_labels = pd.read_csv('../input/data-science-london-scikit-learn/trainLabels.csv',header = None)
test_data =  pd.read_csv('../input/data-science-london-scikit-learn/test.csv',header = None)

a=[train_data,train_labels,test_data]
for i in a :
    print (i.shape)
    

train_data.info()
train_data.describe()
train_data=train_data/17
X, y = train_data, np.ravel(train_labels)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

n_neig=np.arange(2,20)
acc=[]
for i in n_neig:
    KNNModel = KNeighborsClassifier(n_neighbors= i)
    KNNModel.fit(X, y)
    acc.append(KNNModel.score(X, y))
    
plt.plot (n_neig,acc)
plt.xlabel('neighbours')
plt.xlim(2,20)
plt.ylabel('accuracy')
plt.show()
KNNModel = KNeighborsClassifier(n_neighbors= 4)
KNNModel.fit(X, y)
CrossValidateScore = cross_val_score(KNNModel, X, y, cv=5)


# Showing Results
print('Cross Validate Score for Training Set: \n', CrossValidateScore)

submission = pd.DataFrame(KNNModel.predict(test_data))
print(submission.shape)
submission.columns = ['Solution']
submission['Id'] = np.arange(1,submission.shape[0]+1)
submission = submission[['Id', 'Solution']]
submission
