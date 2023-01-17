# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.
train_dataset = pd.read_csv("../input/train.csv")
test_dataset = pd.read_csv('../input/test.csv')
X_train = train_dataset.iloc[:,1:].values
y_train = train_dataset.iloc[:, 0].values
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, criterion='entropy')
rfc.fit(X_train, y_train)
row = pd.DataFrame({'ImageId': range(1,len(test_dataset)+1)})
X_test = test_dataset.iloc[:,:].values

y_pred = rfc.predict(X_test)
for i in range(1, 11):
    plt.subplot(2, 5, i)
    image = X_test[i-1].reshape((28,28))
    plt.title(y_pred[i-1])
    plt.yticks([],[])
    plt.xticks([],[])
    plt.imshow(255-image, cmap='gray')
plt.show()
results_RF = row.assign( Label = y_pred ) 
results_RF.to_csv("rf_submission.csv", index=False)
