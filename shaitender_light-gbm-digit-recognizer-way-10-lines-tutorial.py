import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 
#read the data set

digits_train = pd.read_csv("../input/train.csv")

digits_test = pd.read_csv("../input/test.csv")

sample = pd.read_csv('../input/sample_submission.csv')
#head

digits_train.head()

digits_test.head()
four = digits_train.iloc[3,1:]

four.shape


four= four.values.reshape(28,28)

plt.imshow(four,cmap='gray')
#visuallise the array

print(four[5:-5,5:-5])
#avearage values/distributions of features

description = digits_train.describe()

description
num_class = len(digits_train.iloc[:,0].unique())
x_train= digits_train.iloc[:,1:]

y_train=digits_train.iloc[:,0]



x_test = digits_test.values

y_test=digits_test.iloc[:,0]



#rescaling the feature

from sklearn.preprocessing import scale

x_train = scale(x_train)

x_test=scale(x_test)



#print

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
import lightgbm as lgb

print ('Training lightgbm')



lgtrain = lgb.Dataset(x_train, y_train)

lgval = lgb.Dataset(x_test, y_test)



# params multiclass

params = {

          "objective" : "multiclass",           

          "max_depth": -1,

           "num_class":num_class,

          "learning_rate" : 0.0001,                 

          "verbosity" : 1 }



model = lgb.train(params, lgtrain, 500, valid_sets=[lgtrain, lgval], early_stopping_rounds=750, verbose_eval=200)
# predict results

results = model.predict(x_test)



# select the index's with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("submission.csv",index=False)
submission.head()