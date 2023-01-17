# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "-ltrh", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.





# coding: utf-8



# In[8]:





from numpy import loadtxt

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, average_precision_score





# In[9]:





# load data

lines = open('../input/creditcard.csv').readlines()[1:]



lines2 = []

for l in lines:

    lines2.append( l.replace('"', '') )



print( lines2[0] )

dataset = np.loadtxt(lines2, delimiter=',')



# In[10]:





# split data into X and y

X = dataset[:,0:-1]

Y = dataset[:,-1]





# In[11]:





X[0:5]





# In[12]:





# split data into train and test sets

seed = 10

test_size = 0.33

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)





# In[13]:





# fit model no training data

model = XGBClassifier()

model.fit(X_train, y_train)





# In[34]:





X_test_new = []



for i,val in enumerate(y_test):

    if val == 1.0:

        X_test_new.append( X_test[i])



y_test_new = [1.0]*len(X_test_new)





# In[35]:





# make predictions for test data 

y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]



y_pred_new = model.predict(X_test_new)

predictions_new = [round(value) for value in y_pred_new]





# In[36]:



# y_pred_new



# In[37]:

file = open("submission.csv","w")

for pred,x in zip(y_pred,X_test):

    file.write(",".join([str(int(x[0])),str(pred)]))

    file.write("\n")

    





# In[38]:





# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))



accuracy = accuracy_score(y_test_new, predictions_new)

print("Accuracy: %.2f%%" % (accuracy * 100.0))



# In[19]:

auprc = average_precision_score(y_test,predictions)

print("AU precision-recall curve: ",auprc)



auprc = average_precision_score(y_test_new,predictions_new)

print("AU precision-recall curve: ",auprc)



# In[20]:



sum(y_test)/len(y_test)




