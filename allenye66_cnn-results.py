# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import numpy as np

import keras

import tensorflow as tf

import sklearn.metrics as metrics

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels

import seaborn as sns

import pickle

from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score, f1_score

import os

from collections import Counter

from keras.models import model_from_json

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
json_file = open('/kaggle/input/bagged-malware-cnn/bagged_cnn_0.json', 'r')

loaded_model_json = json_file.read()

json_file.close()



loaded_model = tf.keras.models.model_from_json(loaded_model_json)

loaded_model.load_weights("/kaggle/input/bagged-malware-cnn/bagged_cnn_0.h5")





df = pd.read_csv('/kaggle/input/final-opcodes/all_data.csv')

family_count = [162, 184, 986, 332, 156, 873, 597, 553, 129, 158, 210, 532, 153, 180, 406, 346, 937, 929, 762, 837, 303]





df = df.loc[:, df.columns != 'Total Opcodes']

df = df.loc[:, df.columns != 'File Name']



labels = np.asarray(df[['Family']].copy())



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

labels = le.fit_transform(df['Family'])



for i in range(31):

    df = df.drop(df.columns[1], axis=1)



test_size = 0.2





opcode_sequence = (df.drop(df.columns[0], axis=1))

X_train, X_test, y_train, y_test = train_test_split(opcode_sequence, labels, test_size=test_size, random_state=42)





X_test = tf.reshape(X_test, (1945, 1000, 1))

mapping = Counter(y_test)

#print(Counter(y_test))

mapping = dict(sorted(mapping.items()))
preds = loaded_model.predict_classes(X_test)

print("Accuracy = {}".format(accuracy_score(y_test, preds)))

print("Balanced Accuracy = {}".format(balanced_accuracy_score(y_test, preds)))

print("Precision = {}".format(precision_score(y_test, preds, average='macro')))

print("Recall = {}".format(recall_score(y_test, preds, average='macro')))

print("F1 = {}".format(f1_score(y_test, preds, average='weighted')))


def write_cm(cm):

	file = open("cm_nn.txt","w")

	for y in range(0, 21):

		for x in range(0, 21):

			string = (str(x) + " " + str(y) + " "+ str(round(cm[y][x],4)))

			file.write(string + "\n")





	file.close()
label_map = {"0":"ADLOAD","1":"AGENT","2":"ALLAPLE_A","3":"BHO","4":"BIFROSE","5":"CEEINJECT","6":"CYCBOT_G","7":"FAKEREAN","8":"HOTBAR","9":"INJECTOR","10":"ONLINEGAMES","11":"RENOS","12":"RIMECUD_A","13":"SMALL","14":"TOGA_RFN","15":"VB","16":"VBINJECT","17":"VOBFUS", "18":"VUNDO","19":"WINWEBSEC","20":"ZBOT"  }



def plot_confusion_matrix(y_true,y_predicted):

	cm = metrics.confusion_matrix(y_true, y_predicted)

	l = list(cm)

	#print(l)



	s = 0



	for array in l:

		for value in array:

			s += value



	ooga = []

	counter = 0

	for array in l:

		array = list(array)

		array = [round(x /mapping[counter],3)  for x in array]

		ooga.append(array)

		counter += 1



	write_cm(ooga)

	labels = list(label_map.values())





	df_cm = pd.DataFrame(ooga,index = labels,columns = labels)

	plt.figure(figsize=(20,10)) 



	ax = sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')

	print(type(ax))

	plt.yticks([0.5,1.5,2.5,3.5,4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5], labels,va='center')

	plt.ylabel('True label')

	plt.xlabel('Predicted label')

 

	plt.show()

	plt.close()



plot_confusion_matrix(y_test, preds)