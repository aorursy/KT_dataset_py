# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")

df.iloc[:,1:] /= 255

df.sample(2)

y = df.iloc[:,:1]

x = df.iloc[:,1:]
from math import ceil



n = x.shape[0]

n_train = ceil(0.8 * n)

n_test = ceil(0.2 * n)



x_train = x[:n_train]

y_train = y[:n_train]



x_test = x[n_train:]

y_test = y[n_train:]



y_train = np.squeeze(y_train)

y_test = np.squeeze(y_test)
from sklearn.neural_network import MLPClassifier



mlp_model = MLPClassifier(solver='adam', 

                          alpha=1e-5, 

                          hidden_layer_sizes=(1024, 128), 

                          random_state=1, 

                          max_iter=150)



print(mlp_model)



mlp_model.fit(x_train, y_train)                         

pred_mlp =  mlp_model.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn import metrics



print("\n\nTesting using MLP")

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred_mlp))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred_mlp))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred_mlp))) 
print("\n\nMLP")

print("--- confusion_matrix ---")

print(confusion_matrix(y_test, pred_mlp))  

print("\n--- classification report ---")

print(classification_report(y_test, pred_mlp))  

print("\nmodel accuracy: ", accuracy_score(y_test, pred_mlp))
df = pd.read_csv("../input/test.csv")

df.iloc[:,1:] /= 255

df.sample(2)
test = df.values

test
preds =  mlp_model.predict(test)
index = df.index + 1

d = {"ImageId" : index, "Label" : preds}

df_out = pd.DataFrame(data=d)

df_out.sample(1)
df_out.to_csv("rec_numbers.csv", index=False)