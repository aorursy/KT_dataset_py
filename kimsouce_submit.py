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
from tensorflow.keras.optimizers import Adam





import numpy as np

import pandas as pd

import tensorflow as tf



from sklearn import model_selection
train = pd.read_csv("../input/pubg-finish-placement-prediction/train_V2.csv")

train = train.dropna() #nan값 drop시켜줌

train.shape
test = pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")

test = test.dropna() #nan값 drop시켜줌

test.shape
feature_name = ['boosts','damageDealt', 'heals', 'kills','longestKill', 'walkDistance', 'weaponsAcquired']

label_name = ["winPlacePerc"]
x_train = train[feature_name]

x_test = test[feature_name]

y_train = train[label_name]
from sklearn.linear_model import LinearRegression



model = LinearRegression()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

y_pred.shape
submission = pd.read_csv("../input/pubg-finish-placement-prediction/sample_submission_V2.csv", index_col="Id")

submission.head()
submission["winPlacePerc"] = y_pred
submission.head
submission.to_csv("/kaggle/working/submission.csv")