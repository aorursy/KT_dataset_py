%pylab inline

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Embedding, LSTM



from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.utils import resample

from sklearn.model_selection import cross_val_score



from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.optimizers import SGD, RMSprop

from keras.regularizers import l2

from keras.backend.tensorflow_backend import set_session

from keras.utils import np_utils

import keras



import tensorflow as tf

import pandas as pd

import numpy as np



from IPython.display import SVG, Image



np.random.seed(0)
data = pd.read_csv('../input/creditcard.csv')



# Preprocessing Amount

amt_scale = StandardScaler()

data['NormAmount'] =  amt_scale.fit_transform(data['Amount'].values.reshape(-1, 1))



# Split Train and Test Data

X = data.drop(['Time', 'Amount', 'Class'], axis=1).as_matrix()

Y = data['Class'].as_matrix()



# Standardization

scale_x = StandardScaler()

X = scale_x.fit_transform(X)



train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.25, random_state=1)



fraud_test_y = test_y == 1

fraud_test_x = test_x[fraud_test_y]

fraud_test_y = test_y[fraud_test_y]



train_category_y = np_utils.to_categorical(train_y)

test_category_y = np_utils.to_categorical(test_y)
print('The number of Fraud transactions in Training Data:', train_y[train_y == 1].shape[0])

print('The number of Fraud transactions in Test Data:',  test_y[test_y == 1].shape[0])
pd.value_counts(data['Class'], sort=True)
def resample(X, Y):

    index = np.arange(Y.shape[0])

    fraud_indices = index[Y == 1]

    normal_indices = index[Y == 0]

    random_normal_indices = np.random.choice(normal_indices, len(fraud_indices))

    

    sample_indices = np.concatenate([fraud_indices, random_normal_indices])

    np.random.shuffle(sample_indices)

    sample_indices = np.array(sample_indices)

    

    sample_x = X[sample_indices]

    sample_y = Y[sample_indices]

    return sample_x, sample_y
lg = LogisticRegression()

lg.fit(train_x, train_y)

predicted_y = lg.predict(test_x)

accuracy_score(test_y, predicted_y)
predicted_y = lg.predict(fraud_test_x)

accuracy_score(fraud_test_y, predicted_y)
lg = LogisticRegression()

lg.fit(*resample(train_x, train_y))



predicted_y = lg.predict(test_x)

accuracy_score(test_y, predicted_y)
predicted_y = lg.predict(fraud_test_x)

accuracy_score(fraud_test_y, predicted_y)
dtc = DecisionTreeClassifier(max_depth=10, criterion='entropy')

dtc.fit(train_x, train_y)

predicted_y = dtc.predict(test_x)

accuracy_score(test_y, predicted_y)
predicted_y = dtc.predict(fraud_test_x)

accuracy_score(fraud_test_y, predicted_y)
dtc = DecisionTreeClassifier(max_depth=10, criterion='entropy')

for i in range(3):

    dtc.fit(*resample(train_x, train_y))

    predicted_y = dtc.predict(test_x)

    print(accuracy_score(test_y, predicted_y))
predicted_y = dtc.predict(fraud_test_x)

accuracy_score(fraud_test_y, predicted_y)
config = tf.ConfigProto()

config.gpu_options.per_process_gpu_memory_fraction = 0.1

set_session(tf.Session(config=config))
def generate_model():

    model = Sequential()

    model.add(Dense(output_dim=512, input_dim=29, name='dense01'))

    model.add(Activation('sigmoid', name='activation01'))

    model.add(Dropout(0.5))

    model.add(Dense(output_dim=256, input_dim=512, name='dense02'))

    model.add(Activation('sigmoid', name='activation02'))

    model.add(Dropout(0.5))

    model.add(Dense(output_dim=1, name='dense03'))

    model.add(Activation('sigmoid', name='activation03'))



    model.compile(loss='binary_crossentropy', 

                  optimizer='adam', 

                  metrics=['accuracy'])

    return model





# # Visualization

model = generate_model()

model.summary()
model = generate_model()

model.fit(train_x, train_y, verbose=2)
predicted_y = model.predict(test_x)

predicted_y = predicted_y.reshape(predicted_y.shape[0])

predicted_y = np.where(predicted_y >= 0.5, 1, 0)

print(accuracy_score(test_y, predicted_y))
predicted_y = model.predict(fraud_test_x)

predicted_y = predicted_y.reshape(predicted_y.shape[0])

predicted_y = np.where(predicted_y >= 0.5, 1, 0)

accuracy_score(fraud_test_y, predicted_y)
model = generate_model()



for i in range(20):

    history = model.fit(*resample(train_x, train_y), verbose=0, nb_epoch=10)

    loss = np.mean(history.history.get('loss'))

    acc = np.mean(history.history.get('acc'))

    print('Epoch[%2d]' % i, 'loss: %.4f, ' % loss, 'Accuracy: %.4f'% acc)

    
predicted_y = model.predict(test_x)

predicted_y = predicted_y.reshape(predicted_y.shape[0])

predicted_y = np.where(predicted_y >= 0.5, 1, 0)

print(accuracy_score(test_y, predicted_y))
predicted_y = model.predict(fraud_test_x)

predicted_y = predicted_y.reshape(predicted_y.shape[0])

predicted_y = np.where(predicted_y >= 0.5, 1, 0)

accuracy_score(fraud_test_y, predicted_y)