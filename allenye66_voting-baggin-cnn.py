import numpy as np 
import pandas as pd 
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from collections import Counter
import keras
from keras.models import Sequential 
from keras.layers import Activation, MaxPooling1D, Dropout, Flatten, Reshape, Dense, Conv1D, LSTM,SpatialDropout1D
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from random import randrange
from random import seed
from random import random
import pickle
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import VotingClassifier
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import time
from xgboost import XGBClassifier

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
json_file = open('/kaggle/input/bagged-malware-cnn/bagged_cnn_0.json', 'r')
mj0 = json_file.read()
json_file.close()

m0 = tf.keras.models.model_from_json(mj0)
m0.load_weights("/kaggle/input/bagged-malware-cnn/bagged_cnn_0.h5")
json_file = open('/kaggle/input/bagged-malware-cnn/bagged_cnn_1.json', 'r')
mj1 = json_file.read()
json_file.close()

m1 = tf.keras.models.model_from_json(mj1)
m1.load_weights("/kaggle/input/bagged-malware-cnn/bagged_cnn_1.h5")
json_file = open('/kaggle/input/bagged-malware-cnn/bagged_cnn_2.json', 'r')
mj2 = json_file.read()
json_file.close()

m2 = tf.keras.models.model_from_json(mj2)
m2.load_weights("/kaggle/input/bagged-malware-cnn/bagged_cnn_2.h5")
json_file = open('/kaggle/input/bagged-malware-cnn/bagged_cnn_3.json', 'r')
mj3 = json_file.read()
json_file.close()

m3 = tf.keras.models.model_from_json(mj3)
m3.load_weights("/kaggle/input/bagged-malware-cnn/bagged_cnn_3.h5")
json_file = open('/kaggle/input/bagged-malware-cnn/bagged_cnn_4.json', 'r')
mj4 = json_file.read()
json_file.close()

m4 = tf.keras.models.model_from_json(mj4)
m4.load_weights("/kaggle/input/bagged-malware-cnn/bagged_cnn_4.h5")
json_file = open('/kaggle/input/bagged-malware-cnn/bagged_cnn_5.json', 'r')
mj5 = json_file.read()
json_file.close()

m5 = tf.keras.models.model_from_json(mj5)
m5.load_weights("/kaggle/input/bagged-malware-cnn/bagged_cnn_5.h5")
json_file = open('/kaggle/input/bagged-malware-cnn/bagged_cnn_6.json', 'r')
mj6 = json_file.read()
json_file.close()

m6 = tf.keras.models.model_from_json(mj6)
m6.load_weights("/kaggle/input/bagged-malware-cnn/bagged_cnn_6.h5")
json_file = open('/kaggle/input/bagged-malware-cnn/bagged_cnn_7.json', 'r')
mj7 = json_file.read()
json_file.close()

m7 = tf.keras.models.model_from_json(mj7)
m7.load_weights("/kaggle/input/bagged-malware-cnn/bagged_cnn_7.h5")
json_file = open('/kaggle/input/bagged-malware-cnn/bagged_cnn_8.json', 'r')
mj8 = json_file.read()
json_file.close()

m8 = tf.keras.models.model_from_json(mj8)
m8.load_weights("/kaggle/input/bagged-malware-cnn/bagged_cnn_8.h5")
json_file = open('/kaggle/input/bagged-malware-cnn/bagged_cnn_9.json', 'r')
mj9 = json_file.read()
json_file.close()

m9 = tf.keras.models.model_from_json(mj9)
m9.load_weights("/kaggle/input/bagged-malware-cnn/bagged_cnn_9.h5")

df = pd.read_csv('/kaggle/input/final-opcodes/all_data.csv')


df = df.loc[:, df.columns != 'Total Opcodes']
df = df.loc[:, df.columns != 'File Name']

labels = np.asarray(df[['Family']].copy())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(labels)

for i in range(31):
    df = df.drop(df.columns[1], axis=1)



opcode_sequence = (df.drop(df.columns[0], axis=1))
X_train, X_test, y_train, y_test = train_test_split(opcode_sequence, labels, test_size=0.1, random_state=42)


