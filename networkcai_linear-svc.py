# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv") #Input data files are available in the "../input/" directory.

y_train = train_data[["label"]]

x_train = train_data.drop(["label"], axis=1)





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))









# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC



ss = StandardScaler()

x_train = ss.fit_transform(x_train)

x_test = ss.fit_transform(test_data)



lsvc=LinearSVC()

lsvc.fit(x_train,y_train)

predictions = lsvc.predict(x_test)

counter = np.arange(1,28001)

c1 = pd.DataFrame({'ImageId': counter})

c2 = pd.DataFrame({'Label':predictions})

res = pd.concat([c1, c2], axis = 1)

res.to_csv('output.csv', index = False)