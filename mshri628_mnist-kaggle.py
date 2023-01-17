# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import timeit
from sklearn.tree import DecisionTreeClassifier
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Read all the csv files
test_df=pd.read_csv("../input/test.csv")
test_data=test_df.values
df=pd.read_csv("../input/train.csv")
data=df.values
sam_sub_df=pd.read_csv("../input/sample_submission.csv")
sam_sub_data=sam_sub_df.values
#train and test data
print("TRAIN DATA")
print(data.shape)
print(data)
print("TEST DATA")
print(test_data.shape)
print(test_data)
print("SAMPLE SUBMISSION DATA")
print(sam_sub_data.shape)
print(sam_sub_data)
#split the train data in two parts 
#x_train and x_test
#By x_test we will get the accuracy
print("data=\n",data,data.shape)
m,n=data.shape
x_train=data[0:30000,1:n+1]#fisrt 30,000 rows [0-29999] with all coloums except first
y_train=data[0:30000,0:1]#fisrt 30,000 rows [0-29999] with first coloum
x_test=data[30000:m+1,1:n+1]#left 12000 rows [30000-41999] with all coloums except first
y_test=data[30000:m+1,0:1]#left 12000 rows [30000-41999] with first coloum
print("x_train=\n",x_train,x_train.shape)
print("y_train=\n",y_train,y_train.shape)
print("x_test=\n",x_test,x_test.shape)
print("y_test=\n",y_test,y_test.shape)
#train the classifier
start = timeit.default_timer()
classifier=DecisionTreeClassifier()
classifier.fit(x_train,y_train)
stop = timeit.default_timer()
print('TRAINING TIME for traning  ',x_train.shape[0]," images is=", stop - start)
#check the prediction for the first four rows
tst=classifier.predict(x_train[0:4,:])
print(tst)
print(y_train[0:4])

#calculate the accuracy
start=timeit.default_timer()
y_predicted=classifier.predict(x_test)
count=0
for i in range(x_test.shape[0]):
    if y_predicted[i]==y_test[i]:
        count=count+1
    
accuracy=(count/x_test.shape[0])*100
print("accuracy=",accuracy)

stop=timeit.default_timer()
print("Testing time=",stop-start)
#predict lables for test_data
start=timeit.default_timer()
predicted_lable=classifier.predict(test_data)
stop=timeit.default_timer()
print("Testing time for testing",test_data.shape[0]," images=",stop-start)
#make a array for index of csv
image_id=[]
for i in range(1,test_data.shape[0]+1,1):
    image_id.append(i)
#make a dictinary and then convert to csv
d={'ImageId':image_id,'Label':predicted_lable}
answer=pd.DataFrame(d)
answer.to_csv('answer.csv',index=False)
answer.head()