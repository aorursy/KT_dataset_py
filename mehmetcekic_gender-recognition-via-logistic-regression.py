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
data = pd.read_csv("../input/voice.csv")
data.head(10)
y = data.label
x = data.drop(["label"], axis=1)
# Now we have x data and y data 
# After that x will be trained
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=6)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)

print("test accuracy {}".format(lr.score(x_test,y_test)))

data.label.head(10)
sample_data = data.head(10)
sample_data = sample_data.drop(["label"],axis=1)
y_head = lr.predict(sample_data)
print(y_head)
bigger_sample = data.iloc[100:200,]
#original values
y_original = bigger_sample["label"].values

#prediction
bigger_sample = bigger_sample.drop(["label"],axis=1)
y_head = lr.predict(bigger_sample)

x_comp = np.array(y_original)
y_comp = np.array(y_head)
print(np.array((x_comp,y_comp)).T)

# Let's see what is the worng predicted data more professionally
counter = 0
for i in range(len(x_comp)):
    if x_comp[i] != y_comp[i]:
        counter = counter +1
        print("original data is {} but predicted data is {}.".format(x_comp[i],y_comp[i]))
print("there is only {} wrong prediction with this sample of data".format(counter))