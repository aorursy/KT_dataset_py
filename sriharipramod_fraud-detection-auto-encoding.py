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
import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split



from keras.layers import Input, Dense, Dropout

from keras.models import Model



import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv("../input/Fraud_data_amtstd.csv")
data.shape
data.head()
data.columns
data.dtypes
print(pd.value_counts(data['Class']))



print(pd.value_counts(data['Class'])/data['Class'].shape[0])
# Drawing a barplot

pd.value_counts(data['Class']).plot(kind = 'bar', rot=0)



# Giving titles and labels to the plot

plt.title("Transaction class distribution")

plt.xticks(range(2), ["Normal", "Fraud"])

plt.xlabel("Class")

plt.ylabel("Frequency");
data = data.values
data_nf = data[data[:,-1] == 0]

test_f  = data[data[:,-1] == 1]



train_nf, test_nf = train_test_split(data_nf, test_size=0.2, random_state=123)
print(data.shape)

print(train_nf.shape)

print(test_nf.shape)

print(test_f.shape)
print(np.unique(data[:,-1], return_counts=True))

print(np.unique(train_nf[:,-1], return_counts=True))

print(np.unique(test_nf[:,-1], return_counts=True))

print(np.unique(test_f[:,-1], return_counts=True))
X_train_nf = train_nf[:,:-1]



X_test_nf = test_nf[:,:-1]



X_test_f = test_f[:,:-1]
input_dim = X_train_nf.shape[1]

#encoding_dim = 15

encoding_dim = 150
# Input placeholder

input_att = Input(shape=(input_dim,))



input_dropout = Dropout(0.2)(input_att)

 

# "encoded" is the encoded representation of the input

encoded = Dense(encoding_dim, activation='relu')(input_dropout)



# "decoded" is the lossy reconstruction of the input

decoded = Dense(input_dim, activation='linear')(encoded)
autoencoder = Model(input_att, decoded)
autoencoder.compile(loss='mean_squared_error', optimizer='adam')
%time autoencoder.fit(X_train_nf, X_train_nf, epochs=50, shuffle=True, validation_split=0.2, verbose=1)
autoencoder.evaluate(X_train_nf, X_train_nf)
autoencoder.evaluate(X_test_nf, X_test_nf)
autoencoder.evaluate(X_test_f, X_test_f)
def mse_for_each_record(act, pred):

    error = act - pred

    squared_error = np.square(error)

    mean_squared_error = np.mean(squared_error, axis=1)

    return mean_squared_error
pred_train_nf = autoencoder.predict(X_train_nf)



mse_train_nf = mse_for_each_record(X_train_nf, pred_train_nf)
pred_test_nf = autoencoder.predict(X_test_nf)



mse_test_nf = mse_for_each_record(X_test_nf, pred_test_nf)
pred_test_f = autoencoder.predict(X_test_f)



mse_test_f = mse_for_each_record(X_test_f, pred_test_f)
plt.subplot(1, 3, 1)

plt.boxplot(mse_train_nf)



plt.subplot(1, 3, 2)

plt.boxplot(mse_test_nf)



plt.subplot(1, 3, 3)

plt.boxplot(mse_test_f)
print("-------mse_train_nf-------")

print(pd.Series(mse_train_nf).describe())

print("\n-------mse_test_NF-------")

print(pd.Series(mse_test_nf).describe())

print("\n-------mse_test_f-------")

print(pd.Series(mse_test_f).describe())
cut_off = np.round(np.percentile(mse_train_nf,99),2)



print("Cut-off = {}".format(cut_off))
print("Non-fraud train records = {}%".format(np.round(np.sum(mse_train_nf <= cut_off)/train_nf.shape[0],2)*100))

print("Non-fraud test records  = {}%".format(np.round(np.sum(mse_test_nf <= cut_off)/test_nf.shape[0],2)*100))

print("Fraud test records      = {}%".format(np.round(np.sum(mse_test_f > cut_off)/test_f.shape[0],2)*100))