import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import tensorflow as tf

tf.__version__



from sklearn.metrics import classification_report, confusion_matrix
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

sample_submission=pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')



X_train=train.drop(columns=['label']).values

y_train=train.label.values

#Normalize the data

X_train=tf.keras.utils.normalize(X_train, axis=1)

X_test=tf.keras.utils.normalize(test, axis=1).values



print(X_train.shape, y_train.shape, X_test.shape)



X_test1 = X_test.reshape(X_test.shape[0],28,28,1)

X_train1 = X_train.reshape(X_train.shape[0],28,28,1)
new_model=tf.keras.models.load_model('/kaggle/input/digit-recognizer-98-8-out-sample-accuracy/my_digit_recognizer')
y_pred_train=new_model.predict_classes(X_train1)

y_pred=new_model.predict_classes(X_test1)
cm=confusion_matrix(y_train,y_pred_train)

cm=pd.DataFrame(cm, index=[i for i in range(10)], columns=[i for i in range(10)])

plt.figure(figsize=(10,10))

sns.heatmap(cm, cmap='Blues',linecolor='black',linewidths=1,annot=True,fmt='')
X_test_1=X_test.reshape(X_test.shape[0],28,28)

plt.imshow(X_test_1[100])

plt.show()

print('Prediction: ', y_pred[100])