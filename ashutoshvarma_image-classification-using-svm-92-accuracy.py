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

        os.path.join(dirname, filename)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import cv2

import os

from tqdm import tqdm
DATADIR = '../input/color-classification/ColorClassification'

CATEGORIES = ['orange','Violet','red','Blue','Green','Black','Brown','White']

IMG_SIZE=100
for category in CATEGORIES:

    path=os.path.join(DATADIR, category)

    for img in os.listdir(path):

        img_array=cv2.imread(os.path.join(path,img))

        plt.imshow(img_array)

        plt.show()

        break

    break

training_data=[]

def create_training_data():

    for category in CATEGORIES:

        path=os.path.join(DATADIR, category)

        class_num=CATEGORIES.index(category)

        for img in os.listdir(path):

            try:

                img_array=cv2.imread(os.path.join(path,img))

                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))

                training_data.append([new_array,class_num])

            except Exception as e:

                pass

create_training_data()            
print(len(training_data))
lenofimage = len(training_data)
X=[]

y=[]



for categories, label in training_data:

    X.append(categories)

    y.append(label)

X= np.array(X).reshape(lenofimage,-1)

##X = tf.keras.utils.normalize(X, axis = 1)

X.shape
X = X/255.0
X[1]
y=np.array(y)
y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y)



from sklearn.svm import SVC

svc = SVC(kernel='linear',gamma='auto')

svc.fit(X_train, y_train)
y2 = svc.predict(X_test)
from sklearn.metrics import accuracy_score

print("Accuracy on unknown data is",accuracy_score(y_test,y2))
from sklearn.metrics import classification_report

print("Accuracy on unknown data is",classification_report(y_test,y2))
result = pd.DataFrame({'original' : y_test,'predicted' : y2})
result
