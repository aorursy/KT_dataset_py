# import libraries & parkages

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import keras # to build CNN

from keras.utils.np_utils import to_categorical



import datetime
df_train = pd.read_csv('../input/train.csv')

X_train = df_train.drop(labels='label',axis=1)

Y_train = df_train['label'] # save the 'label' column in the train dataset

del df_train # delete unsed dataset to save memory



X_test = pd.read_csv('../input/test.csv')



print(X_train.shape, Y_train.shape, X_test.shape)

X_train.head()
sns.countplot(Y_train)
X_train.isnull().any().describe()
Y_train.isnull().any()
X_test.isnull().any().describe()
# CNN is easier to coverge in [0,1]

X_train = X_train/255

X_test = X_test/255
#X_train = X_train.values.reshape(-1,28,28,1) # 28x28 image

#X_test = X_test.values.reshape(-1,28,28,1)

Y_train = to_categorical(Y_train, num_classes=10)

print(X_train.shape, Y_train.shape, X_test.shape)
xx_train, xx_validation, yy_train, yy_validation = train_test_split(X_train,Y_train,test_size=0.2, random_state=42)

print(xx_train.shape, yy_train.shape, xx_validation.shape, yy_validation.shape)
plt_random = np.random.randint(0,33600-1)

print('the label for #%d-th image is %s' % (plt_random, str(yy_train[plt_random,:])))

plt.imshow(xx_train[plt_random][:,:,0])
type(xx_train)
type(yy_train)
from keras.models import Sequential

from keras.layers import Dense,Convolution2D,MaxPool2D, Activation

from keras.optimizers import RMSprop



#build the CNN

model = Sequential([

    Dense(32, input_dim=784),

    Activation('relu'),

    Dense(10),

    Activation('softmax')

])



#define the optimizer

rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)



model.compile(

    optimizer=rmsprop,

    loss='categorical_crossentropy',

    metrics=['accuracy']

    )
print('\nTraining ----------------')

model.fit(xx_train.values,yy_train,epochs=20, batch_size=32)
print('\nValidating ----------------')

loss, accuracy = model.evaluate(xx_validation.values,yy_validation)

print('validation loss:', loss)

print('validation accuracy:',accuracy)
Y_pred = model.predict(X_test.values)
# convert back to 0-9

Y_pred = np.argmax(Y_pred,axis=1)
Y_pred.shape
# Save results

result_submission = pd.DataFrame({

    'ImageId': np.arange(1,28001,1),

    'Label': Y_pred

})

result_submission.to_csv('submission_2layer_nn.csv',index=False)