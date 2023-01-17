import os

print(os.listdir('../input/dm-and-pr-ws1920-machine-learning-competition'))
import pandas as pd 

import numpy as np 



from sklearn.tree import DecisionTreeClassifier
PATH = '../input/dm-and-pr-ws1920-machine-learning-competition/'

df_train = pd.read_csv(PATH+'train.csv')

df_test = pd.read_csv(PATH+'test.csv')

sample_sub = pd.read_csv(PATH+'sampleSubmission.csv')
X = df_train.profession.values # this line is awful 

y = df_train.target.values 



X_test = df_test.profession.values
model = DecisionTreeClassifier(max_depth=4)

model.fit(X.reshape(-1,1),y)

y_hat = model.predict_proba(X_test.reshape(-1,1))[:,1]
y_hat
sample_sub['target'] = y_hat 

sample_sub.head()

sample_sub.to_csv('estimation_01.csv', index=False)