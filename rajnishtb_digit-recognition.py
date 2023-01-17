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
import pandas as pd
data = pd.read_csv("../input/train.csv")
data.head()
img = data.iloc[0, 1:].values
img
img = img.reshape(28, 28)
img
import matplotlib.pyplot as plt
plt.imshow(img)
X = data.iloc[: , 1:].values
X
y = data.iloc[: , 0].values
y
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X, y)
test_data = pd.read_csv("../input/test.csv")
result = rfc.predict(test_data)
result
imageId = []
label = []
for a, b in enumerate(result):
    imageId.append(a+1)
    label.append(b)
submit_data = pd.DataFrame({'imageId':imageId, 'label':label})
submit_data
submit_data.to_csv("submit.csv")
