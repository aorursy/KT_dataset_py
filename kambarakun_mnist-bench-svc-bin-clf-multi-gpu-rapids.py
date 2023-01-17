%%time



# https://www.kaggle.com/cdeotte/rapids



import sys

!cp ../input/rapids/rapids.0.12.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
%%time



import os



from pathlib import Path



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



from cuml.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import train_test_split

from tqdm import tqdm
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%%time



path_input = Path('/kaggle/input/digit-recognizer/')



df_train = pd.read_csv(path_input / 'train.csv')

df_test  = pd.read_csv(path_input / 'test.csv')



X_train = df_train.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.float32)

y_train = df_train.iloc[:, 0 ].values



X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=42)



X_test  = df_test.values.reshape(-1, 28, 28).astype(np.float32)
fig = plt.figure(figsize=(18, 9))

for row in range(5):

    for col in range(10):

        ax = fig.add_subplot(5, 10, 1+10*row+col)

        ax.imshow(X_train[10*row+col])

        ax.set_axis_off()

        ax.set_title(y_train[10*row+col])
%%time



list_model = []



for ans in tqdm(range(10)):

    model = SVC()

    model.fit(X_train.reshape(-1, 28*28), y_train==ans)

    list_model.append(model)
%%time



list_h_valid = []



for ans in tqdm(range(10)):

    h_valid = list_model[ans].predict(X_valid.reshape((-1, 28*28)))

    h_valid = h_valid.values.get()

    list_h_valid.append(h_valid)



h_valid = np.array(list_h_valid).argmax(axis=0)
confusion_matrix(y_valid, h_valid)
accuracy_score(y_valid, h_valid)
fig = plt.figure(figsize=(18, 9))

for row in range(5):

    for col in range(10):

        ax = fig.add_subplot(5, 10, 1+10*row+col)

        ax.imshow(X_valid[y_valid != h_valid][10*row+col])

        ax.set_axis_off()

        ax.set_title(h_valid[y_valid != h_valid][10*row+col])