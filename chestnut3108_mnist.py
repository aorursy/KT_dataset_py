# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import sklearn

data = pd.read_csv('../input/train.csv')
a = data.iloc[0,1:].values
a
a = a.reshape(28,28).astype('uint8')
a
%matplotlib inline

plt.imshow(a)
data_x = data.iloc[:,1:]

data_y = data.iloc[:,0]
data_y
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 90)

model.fit(data_x,data_y)
test = pd.read_csv('../input/test.csv')
ab = model.predict(data_x)
var = 0

for i in range(0,42000):

        if ab[i] == data_y[i]:

                             var = var + 1             

                

                
var
ab = model.predict(test)
import caffe
ggle
np.savetxt("file_name.csv", ab, delimiter=",", fmt='%s', header='label')
print(check_output(["ls", "../input"]).decode("utf8"))