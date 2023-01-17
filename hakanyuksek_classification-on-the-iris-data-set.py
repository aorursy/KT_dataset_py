# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv")
data.info()
import matplotlib.pyplot as plt
import seaborn as sns

color = {'Iris-setosa': "r", 'Iris-versicolor': "g", 'Iris-virginica': "b"}
counts = []
kinds = []

for key in color.keys():
    counts.append(len(data[data["species"]==key].values))
    kinds.append(key)
df = pd.DataFrame({"Columns of Data Set":kinds,"# of Samples":counts})
newIndex = (df["# of Samples"].sort_values(ascending = False)).index.values
sortedData = df.reindex(newIndex)

plt.figure(figsize = (10,5))
ax = sns.barplot(x = sortedData["Columns of Data Set"],y = sortedData["# of Samples"])
plt.xticks(rotation = 0)
plt.ylabel("# of Samples")
plt.title("Distribution of the Data according to class label")

data.head(10)
labels = data["species"]
data.drop(["species"],axis=1,inplace=True)
columns = data.columns

for col in columns:
    min = np.array(data[col].values).min()
    max = np.array(data[col].values).max()
    values = []
    for each in data[col]:
        each = (each - min) / (max-min)        
        values.append(each)
    data[col] = values
    
data.head(10)
dict = {}

for each in labels:
    dict[each] = 1
    
classes = dict.keys()

i = 0
for each in classes:
    dict[each] = i
    i += 1
dict
for i in range(len(labels)):
    labels[i] = dict[labels[i]]
# define the keras model
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X = data.drop(["species"],axis = 1,inplace = False)
Y = labels
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.33, random_state=42)

history = model.fit(X_train, y_train,validation_split=0.33,epochs=150, batch_size=10)
# evaluate the keras model
loss, accuracy = model.evaluate(X_test, y_test)
# summarize history for accuracy
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print('Accuracy: %.2f' % (accuracy*100))
# summarize history for loss
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print('Loss: %.2f' % (loss*100))