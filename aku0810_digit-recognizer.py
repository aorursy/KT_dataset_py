import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

train_digits = pd.read_csv('../input/digit-recognizer/train.csv')

test_digits = pd.read_csv('../input/digit-recognizer/test.csv')

sample = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
train_digits.head()
test_digits.head()
fours = train_digits.iloc[3,1:]

fours.shape
fours = fours.values.reshape(28, 28)

plt.imshow(fours, cmap='gray')
print(fours[5:-5,5:-5])
description = train_digits.describe()

description
num_class = len(train_digits.iloc[:,0].unique())
X_train = train_digits.iloc[:,1:]

y_train = train_digits.iloc[:,0]



X_test = test_digits.values

y_test =test_digits.iloc[:,0]



from sklearn.preprocessing import scale

X_train = scale(X_train)

X_test = scale(X_test)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
import lightgbm as lgb

print('Training lightgbm')



lgtrain = lgb.Dataset(X_train,y_train)

lgval = lgb.Dataset(X_test, y_test)



params = {

    "objective": "multiclass",

    "max_depth": -1,

    "num_classes":num_class,

    "learning_rate":0.01,

    

    "verbosity": -1

}

model = lgb.train(params, lgtrain, 500, valid_sets=[lgtrain, lgval], early_stopping_rounds=750, verbose_eval=200)
results = model.predict(X_test)



results = np.argmax(results, axis=1)



results = pd.Series(results, name='Label')



submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results],axis=1)



submission.to_csv("submission.csv", index=False)
submission.head()