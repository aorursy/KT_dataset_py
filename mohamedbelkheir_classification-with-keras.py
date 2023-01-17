%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
data = pd.read_csv("../input/mushrooms.csv")
data.head()
data.info()
data['stalk-root'].value_counts()
100*len(data.loc[data['stalk-root']=='?']) / sum(data['stalk-root'].value_counts())
data = data.drop('stalk-root', 1)
Y = pd.get_dummies(data.iloc[:,0],  drop_first=False)
X = pd.DataFrame()
for each in data.iloc[:,1:].columns:
    dummies = pd.get_dummies(data[each], prefix=each, drop_first=False)
    X = pd.concat([X, dummies], axis=1)
    

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from sklearn.model_selection import cross_val_score
from keras import backend as K
from keras.layers import BatchNormalization
seed = 123456 

def create_model():
    model = Sequential()
    model.add(Dense(20, input_dim=X.shape[1], kernel_initializer='uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    #model.add(Dense(20, input_dim=X.shape[1], kernel_initializer='uniform', activation='relu'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    #sgd = SGD(lr=0.01, momentum=0.7, decay=0, nesterov=False)
    model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])
    return model
model = create_model()
history = model.fit(X.values, Y.values, validation_split=0.50, epochs=100, batch_size=50, verbose=0)



history.history.keys()
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
print("Training accuracy: %.2f%% / Validation accuracy: %.2f%%" % 
      (100*history.history['accuracy'][-1], 100*history.history['val_accuracy'][-1]))
from keras import backend as K
import numpy as np

layer_of_interest=0
intermediate_tensor_function = K.function([model.layers[0].input],[model.layers[layer_of_interest].output])
intermediate_tensor = intermediate_tensor_function([X.iloc[0,:].values.reshape(1,-1)])[0]
intermediates = []
color_intermediates = []
for i in range(len(X)):
    output_class = np.argmax(Y.iloc[i,:].values)
    intermediate_tensor = intermediate_tensor_function([X.iloc[i,:].values.reshape(1,-1)])[0]
    intermediates.append(intermediate_tensor[0])
    if(output_class == 0):
        color_intermediates.append("#0000ff")
    else:
        color_intermediates.append("#ff0000")
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
intermediates_tsne = tsne.fit_transform(intermediates)
