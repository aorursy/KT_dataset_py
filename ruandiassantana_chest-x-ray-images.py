# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2 as cv

from tqdm import tqdm

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

files = []

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if '/chest_xray/chest_xray' not in os.path.join(dirname, filename) and '__MACOSX' not in os.path.join(dirname, filename):

            files.append(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
len(files)
x = np.arange(7500)
df = pd.DataFrame(columns=x,dtype='uint8')
df
for i in tqdm(files):

    img = cv.imread(i)

    img.resize((50, 50, 3),refcheck=False)

    img = img.reshape((7500))

    

    if '/PNEUMONIA/' in i:

        img = np.append(img,1)

    else:

        img = np.append(img,0)

    

    df = df.append(pd.Series(img),ignore_index=True)
df.rename(columns={7500: 'Pneumonia'},inplace=True)
df['Pneumonia'] = df['Pneumonia'].astype(int)
df
df['Pneumonia'].value_counts()
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
X = df.drop('Pneumonia',axis=1)

y = df['Pneumonia']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train/=255

X_test/=255
xgbc = XGBClassifier()
xgbc.fit(X_train,y_train)
xgbc.score(X_test,y_test)
from sklearn.metrics import classification_report,confusion_matrix
predictions = xgbc.predict(X_test)
print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))