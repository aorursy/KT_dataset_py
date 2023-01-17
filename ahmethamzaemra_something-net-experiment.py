# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import os 

os.chdir('../input/')
from something.NNetwork import Fully_Connected
data = pd.read_csv("voicegender/voice.csv")

data.head()
x = data.drop(['label'], axis=1).values

y = data['label'].values

x.shape
from sklearn.preprocessing import StandardScaler

encoder = StandardScaler()

x = encoder.fit_transform(x)
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

y = enc.fit_transform(y)
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=.2, random_state=12)
net = Fully_Connected()
net.init_layers_xavier([20,16,16,2])

net.train(x_train, y_train, x_val=x_val, y_val = y_val, num_iters=10000, verbose=True, learning_rate=0.005) # start training
import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(40, 40))

plt.subplot(5, 5, 1)

plt.plot(net.loss_history)

plt.title("Loss")

plt.subplot(5, 5, 2)

plt.plot(net.train_acc_history, 'b',label='traing accuracy')

plt.plot(net.val_acc_history, 'r', label='validation accuracy')

plt.legend()

plt.title("Accuracy")

plt.show()