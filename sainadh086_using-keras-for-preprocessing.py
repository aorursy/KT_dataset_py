# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = os.listdir("../input/train_images")

data_labels = pd.read_csv('../input/train.csv')
x = data_labels['id_code']

y = data_labels['diagnosis']
X_img = []

y_p = []

def create_training_set(label, path):

    img = cv2.imread(path, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (32,32))

    X_img.append(np.array(img))

    y_p.append(str(label))
from tqdm import tqdm

TRAIN_DIR = '../input/train_images'

for id_code, diagnosis in tqdm(zip(x,y)):

    path = os.path.join(TRAIN_DIR, '{}.png'.format(id_code))

    create_training_set(diagnosis, path)
from keras.utils import to_categorical

#Y = to_categorical(y_p)

Y = np.array(y_p)

X= np.array(X_img)

X=X/255
Y = np.array(y_p)

Y = Y.astype(int)
from sklearn.model_selection import train_test_split

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.20, random_state=0)
X_train = X_train.reshape(2929,1024)

X_test = X_valid.reshape(733,1024)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,Y_train)
y_pred = lr.predict(X_test)

from sklearn.metrics import accuracy_score as ac

ac(Y_valid, y_pred)
from sklearn.metrics import confusion_matrix

def quadratic_kappa(actuals, preds, N):

    w = np.zeros((N,N))

    O = confusion_matrix(actuals, preds)

    for i in range(len(w)): 

        for j in range(len(w)):

            w[i][j] = float(((i-j)**2)/(N-1)**2)

    

    act_hist=np.zeros([N])

    for item in actuals: 

        act_hist[item]+=1

    

    pred_hist=np.zeros([N])

    for item in preds: 

        pred_hist[item]+=1

                         

    E = np.outer(act_hist, pred_hist);

    E = E/E.sum();

    O = O/O.sum();

    

    num=0

    den=0

    for i in range(len(w)):

        for j in range(len(w)):

            num+=w[i][j]*O[i][j]

            den+=w[i][j]*E[i][j]

    return (1 - (num/den))
quadratic_kappa(Y_valid, y_pred,5)
test_df = pd.read_csv('../input/test.csv')

test_df.shape
test_ids = test_df['id_code']
test_images = []

def create_test_set(path):

    img = cv2.imread(path,cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (32,32))

    test_images.append(np.array(img))
for id_code in tqdm(test_ids):

    path = os.path.join('../input/test_images','{}.png'.format(id_code))

    create_test_set(path)
test_x = np.array(test_images)

test_x.shape
test_x = test_x.reshape(1928,32*32)
pred = lr.predict(test_x)

pred
np.unique(pred)
p = pd.DataFrame({'id_code':test_ids,'diagnosis':pred})

p.to_csv('submission.csv',index=False)