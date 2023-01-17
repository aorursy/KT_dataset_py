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
import numpy as np
train_main = pd.read_csv('../input/train.csv')
test_main = pd.read_csv('../input/test.csv')
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
first_digit =train_main.iloc[1].drop('label').values.reshape(28,28)
plt.imshow(first_digit)
train,test=train_test_split(train_main,test_size=0.3,random_state=100)
train_x1=train.drop('label',axis=1)
train_y1=train['label']

test_x1=test.drop('label',axis=1)
test_y1=test['label']


from sklearn.neighbors import KNeighborsClassifier

model_3=KNeighborsClassifier(n_neighbors=5)
model_3.fit(train_x1,train_y1)

pred_test=model_3.predict(test_x1)
accuracy_score(test_y1,pred_test)


test_pred = model_3.predict(test_main)
df_test_pred = pd.DataFrame(test_pred, columns=['Predicted'])
df_test_pred['ImageId'] = test_main.index + 1
df_test_pred[['ImageId', 'Predicted']].to_csv('submission.csv', index=False)
pd.read_csv('submission.csv').head()
