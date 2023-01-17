# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from sklearn.linear_model import Perceptron

import pandas as pd

import numpy as np



data_train = pd.read_table("/kaggle/input/data-to-work-on-basic-examples/class2_tr.txt",header=None)

data_test = pd.read_table("/kaggle/input/data-to-work-on-basic-examples/class2_test.txt",header=None)

x_train = data_train.loc[:,:1]

y_train = data_train.loc[:,2]

x_test = data_test.loc[:,:1]

y_test = data_test.loc[:,2]
model = Perceptron(early_stopping=True,verbose=1)

model.fit(x_train,y_train)

pred = model.predict(x_test)
print("Train Accuracy: ",model.score(x_train,y_train))

print("Test Accuracy:  ",model.score(x_test,y_test))