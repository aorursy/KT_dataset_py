import pandas as pd

import numpy as np

import matplotlib.pyplot as plt





from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn import model_selection
train = pd.read_csv("../input/train.csv")

print(train.shape)



test = pd.read_csv("../input/test.csv")

print(test.shape)
print(train.columns)

train.describe()



print(test.columns)

test.describe()




train_x = train.iloc[:,2:]

train_y_ = train.iloc[:,1]

train_y = []

for i in train_y_:

    train_y.append(i)

    

test_x = test.iloc[:,1:]





#print(test_x)
scaler = StandardScaler()

train_x = scaler.fit_transform(train_x)

#test_x = scaler.fit_transform(test_x)


regression = LogisticRegression(class_weight='balanced', solver='liblinear', penalty ='l1', C= 0.1, max_iter=10000)

regression.fit(train_x,train_y)



scores = model_selection.cross_val_score(regression,train_x,train_y,scoring="accuracy",cv=50)   # To Cross validate and remodel it with less features

# cv - number of runs to find cross validated model



test_y_ = regression.predict_proba(test_x)

print("Training Accuracy score: ",regression.score(train_x,train_y))

test_y = []

#for i in test_y_[:,1]:

#    test_y.append(i)

    

#print(len(train_y))

print((test_y_[:,1]))



#score = accuracy_score(train_y,test_y)

#print(score)

submission = pd.DataFrame({"id":test["id"],"target":test_y_[:,1]})



#Visualize the first 5 rows

submission.head()



filename = 'submission.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)