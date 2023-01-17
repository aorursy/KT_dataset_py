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
from sklearn import datasets

digits = datasets.load_digits()
import pandas as pd



#ファイルの読み込み

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")



print(train.shape)

print(test.shape)
train.head()
test.head()
%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image as im



fig = plt.figure(figsize=(10,5))



#各行を2次元配列に変換

for i in range(50):

    fig.add_subplot(5,10,i+1)

    imgArray = np.array(test.iloc[i:i+1,:])

    b = np.reshape(imgArray, (28, 28))

    pilOUT = im.fromarray(np.uint8(b))

    plt.imshow(pilOUT)



#表示    

plt.show()
from sklearn import ensemble



x_train = train.drop("label", axis=1)

y_train = train["label"]



clf = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2,class_weight="balanced")

clf.fit(x_train, y_train)
y_pred = clf.predict(test)
print(y_pred)

print(y_pred.shape)
imgid = np.array(np.arange(1,28001)).astype(int)

result = pd.DataFrame(y_pred, imgid, columns = ["Label"])

result.to_csv("digits_result.csv", index_label = ["ImageId"])