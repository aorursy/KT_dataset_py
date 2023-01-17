# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import scipy.stats as stats

import seaborn as sns

from sklearn.preprocessing import power_transform, maxabs_scale



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/predicting-a-pulsar-star/pulsar_stars.csv')

data.head()
data.describe()
pulsars = data[data.target_class == 1]

pulsars.describe()
for c in pulsars.columns[:-1]:

    sns.distplot(pulsars[c])

    plt.show()
sns.set()

sns.pairplot(pulsars, size=5)

plt.show();
for c in data.columns[:-1]:

    d = pd.concat([data[c], data['target_class']], axis=1)

    f, ax = plt.subplots(figsize=(8, 6))

    fig = sns.boxplot(x="target_class", y=c, data=d)

    plt.show()
# add more positives

data = data.append([pulsars]*5, ignore_index=True)

data = data.sample(frac=1).reset_index(drop=True)

data.describe()
parsed_data = data.copy()

columns = data.columns[:-1]

parsed_data[columns] = maxabs_scale(power_transform(data.iloc[:,0:8],method='yeo-johnson'))
def plot(preparsed_data, after_parsing, label):

    fig, (ax0, ax1) = plt.subplots(1, 2)

    d = np.sort(preparsed_data.to_numpy())

    fit = stats.norm.pdf(d, np.mean(d), np.std(d))

    ax0.hist(d, density=True)

    ax0.plot(d, fit, '-')

    d2 = np.sort(after_parsing.to_numpy())

    fit2 = stats.norm.pdf(d2, np.mean(d2), np.std(d2))

    ax1.hist(d2, density=True)

    ax1.plot(d2, fit2, '-')

    ax0.set_title(label)
for c in parsed_data.columns[:-1]:

    plot(data[c], parsed_data[c], c)
from keras.layers import Dense, Input

from keras.models import Model

from keras.regularizers import l1
encoder_input = Input(shape=(8,))

encoder_layer = Dense(200, activation='sigmoid', activity_regularizer=l1(10e-5))(encoder_input)
decoder_layer = Dense(8, activation='tanh')(encoder_layer)

autoencoder = Model(inputs=encoder_input, outputs=decoder_layer)

autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

autoencoder.summary()
all_data = parsed_data.drop(['target_class'], axis=1).to_numpy()

train_x, test_x = all_data[:-1000], all_data[-1000:]

train_x.shape
autoencoder.fit(x=train_x, y=train_x, batch_size=8, validation_split=0.3, epochs=10)
autoencoder.evaluate(x=test_x, y=test_x)
encoder =  Model(inputs=encoder_input, outputs=encoder_layer)

encoder.summary()
from sklearn import svm
classifier_test = int(len(data.index) * .3)
encoded_input = encoder.predict(all_data)

train_x, test_x = encoded_input[classifier_test:], encoded_input[:classifier_test]
labels = parsed_data['target_class'].to_numpy()

train_y, test_y =  labels[classifier_test:], labels[:classifier_test]

train_y.shape
final_classifiier = svm.SVC(gamma='scale')

final_classifiier.fit(train_x, train_y)
predicted = final_classifiier.predict(test_x)
from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve



print(classification_report(test_y, predicted))

# Generate ROC curve values: fpr, tpr, thresholds

fpr, tpr, thresholds = roc_curve(test_y, predicted)



# Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()