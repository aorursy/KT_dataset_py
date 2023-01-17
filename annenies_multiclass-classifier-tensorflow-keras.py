# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

import seaborn as sns

train_data = pd.read_csv("../input/fashion-mnist_train.csv")

test_data = pd.read_csv("../input/fashion-mnist_test.csv")

# Store human readable labels in dictionary format so they can be easily mapped

labels = {0 : "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",

          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}

sns.set(style="whitegrid")

train_plt_df = pd.DataFrame(train_data.groupby('label')['pixel1'].count()).reset_index()

ax = sns.barplot(x="label", y="pixel1", data=train_plt_df)

plt.title("Number of Rows by Label in Train Data")
sns.set(style="whitegrid")

test_plt_df = pd.DataFrame(test_data.groupby('label')['pixel1'].count()).reset_index()

ax = sns.barplot(x="label", y="pixel1", data=test_plt_df)

plt.title("Number of Rows by Label in Test Data")
def plot_row(df, row_to_plot):

    plt.figure()

    #Need to reshape dataframe row into a 28x28 numpy array to plot.

    plt.imshow(np.array(df.iloc[row_to_plot, 1:]).reshape(28,28))

    plt.colorbar()

    plt.grid(False)

    plt.title(labels[df.iloc[row_to_plot, 0]])

    plt.show()



plot_row(train_data, 129)
train_data.iloc[:,1:] = train_data.iloc[:,1:]/255

test_data.iloc[:,1:] = test_data.iloc[:,1:]/255
# Tried adding some layers, values for how they changed the outcome are listed. 

model = keras.Sequential([                        #Train/Test = Train-Test

    #keras.layers.Dense(456, activation='sigmoid'), 0.9165/0.8919 = 0.0245

    #keras.layers.Dense(456, activation='relu'), 0.9168/0.8819 = 0.0348

    #keras.layers.Dense(456, activation='tanh'), 0.9036/0.8856 = 0.0179

    #keras.layers.Dense(588, activation='relu'),  0.9136/0.8922 = 0.0213

    keras.layers.Dense(392, activation='relu'), # 0.9147/0.8955 = 0.0191

    keras.layers.Dense(128, activation='relu'), # 0.9088/0.8916 = 0.0172

    keras.layers.Dense(10, activation='softmax')

])
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.fit(train_data.drop('label', axis=1).values, train_data['label'].values, epochs=10)
test_loss, test_acc = model.evaluate(test_data.drop('label', axis=1).values, test_data['label'].values)



print('\nTest accuracy:', test_acc)

predictions = model.predict(test_data.drop('label', axis=1).values)

predictions[0]
labels[np.argmax(predictions[0])]
plot_row(test_data, 0)
res_data = test_data['label'].copy()

res_data = pd.DataFrame(res_data)

res_data['preds'] = [np.argmax(x) for x in predictions]

res_data['wrong'] = res_data['label'] != res_data['preds']

res_data = res_data.groupby(['label', 'preds']).sum().reset_index()

res_data['preds'] =res_data['preds'].map(labels)

res_data['label'] =res_data['label'].map(labels)

res_data.sort_values(by='wrong', ascending=False)[:10]
plt.figure(figsize=(20,10))

sns.set(style="whitegrid")

plt_df = pd.DataFrame((1000-res_data.groupby('label')['wrong'].sum())/1000).reset_index().sort_values(by='wrong')

ax = sns.barplot(x="label", y="wrong", data=plt_df)

plt.title("Accruacy by label")
ax = sns.heatmap(res_data.pivot('label', 'preds', 'wrong'))

plt.title('Heatmap of wrong predictions')
plt_df = test_data[test_data['label'].isin([0, 2, 6])]

plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(np.array(plt_df.iloc[i, 1:]).reshape(28,28), cmap=plt.cm.binary)

    plt.xlabel(labels[plt_df.iloc[i, 0]])

plt.show()
