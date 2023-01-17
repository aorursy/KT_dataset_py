# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#read datasets
train=pd.read_csv("../input/digit-recognizer/train.csv")
test=pd.read_csv("../input/digit-recognizer/test.csv")
#check datasets
print("train type:{0}\ntrain_shape:{1}".format(type(train), train.shape))
print("test type:{0}\ntest_shape:{1}".format(type(test), test.shape))
#split train data to lebal and data
train_label=train["label"]
train_data=train.drop("label", axis=1)

#check each shape
print("train_data:{0}\ntrain_label:{1}".format(train_data.shape, train_label.shape))
#check how one of images is
random_image=train_data.iloc[np.random.randint(10000, 20000)].values.reshape(28, 28)
plt.imshow(random_image, cmap="Greys")
plt.show()
#convert data from 0~255 to 0~1
train_data/=255
test/=255
#let logistic model learn
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(train_data, train_label)
#predict test_label
test_pred=lr.predict(test)
print(test_pred)
#notice that output data is ndarray type
print("input_type:{0}\noutput_type:{1}".format(type(test), type(test_pred)))
submission=pd.DataFrame({"ImageID": np.arange(1, len(test_pred)+1), "Label": test_pred})
submission
submission.to_csv("submission_data.csv", index=False)