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
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
print(train.shape)

print(train.dtypes.value_counts())
print(test.shape)

print(test.dtypes.value_counts())
train.head()
test.head()
train.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)
test.isnull().sum()[test.isnull().sum()>0].sort_values(ascending=False)
import matplotlib.pyplot as plt

from PIL import Image as im



def output_gray_image(df, i):

    img = df.drop(["label"], axis=1).iloc[i].values

    img = img.reshape((28,28))

    plt.imshow(img, cmap='gray')

 

output_gray_image(train, 1)
from sklearn import ensemble



train_x = train.drop("label", axis=1)

train_y = train["label"]



random_forest= ensemble.RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2,class_weight="balanced")

random_forest.fit(train_x, train_y)
y_pred = random_forest.predict(test)
print(y_pred)

print(y_pred.shape)
imgid = np.array(np.arange(1,28001)).astype(int)

result = pd.DataFrame(y_pred, imgid, columns = ["Label"])

result.to_csv("submission.csv", index_label = ["ImageId"])