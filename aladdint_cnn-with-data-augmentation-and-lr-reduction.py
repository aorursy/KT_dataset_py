import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import sklearn

import tensorflow as tf





from sklearn import cluster

from sklearn import metrics

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import scale

from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import AgglomerativeClustering

from sklearn.linear_model import LogisticRegression

from pandas.plotting import scatter_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator







from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score







# load training & test datasets

train = pd.read_csv('../input/Kannada-MNIST/train.csv')

test = pd.read_csv('../input/Kannada-MNIST/test.csv')

train.columns
train.head()
test.head()
#Getting the shape of the pixel data

print( 'Shape of the data is :' , train.loc[2,'pixel0':].shape)



print('Max pixel value:' ,train.loc[2,'pixel0':].max())



print('Min pixel value:' ,train.loc[2,'pixel0':].min())
plt.figure()

plt.imshow(train.loc[2,'pixel0':].values.reshape((28,28)))

plt.colorbar()

plt.grid(False)
plt.figure(figsize=(10,10))

for i in range(25):

	plt.subplot(5,5,i+1)

	plt.xticks([])

	plt.yticks([])

	plt.grid(False)

	plt.imshow(train.loc[i,'pixel0':].values.reshape((28,28)) ,  cmap=plt.cm.binary)

	plt.xlabel(train.loc[i,'label'])
from sklearn.manifold import TSNE



# Sample from the training set

sample_size = 8000



np.random.seed(2018)

idx = np.random.choice(60000, size=sample_size, replace=False)

train_sample = train.loc[idx,'pixel0':]

label_sample = train.loc[idx,'label']



train_sample

# Generate 2D embedding with TSNE

embeddings = TSNE(verbose=2).fit_transform(train_sample)


# Visualize TSNE embedding

vis_x = embeddings[:, 0]

vis_y = embeddings[:, 1]



plt.figure(figsize=(10,7))

plt.scatter(vis_x, vis_y, c=label_sample, cmap=plt.cm.get_cmap("jet", 10), marker='.')

plt.colorbar(ticks=range(10))

plt.clim(-0.5, 9.5)

plt.show()
y_train = train['label']



y_train
train['label'].value_counts()
g = sns.countplot(y_train)



y_train.value_counts()
import pandas as pd

import numpy as np

import seaborn as sns

%matplotlib inline



import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical



np.random.seed(2)



from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import confusion_matrix

import itertools



sns.set(style='white', context='notebook', palette='deep')
train
test
#Seperating all the features and target for training data

train_data = train.loc[:,'pixel0':]

train_label = train.loc[:,'label']

print(f"train_data shape :{train_data.shape}")

print(f"train_label shape :{train_label.shape}")
#Normalize

X = train_data.values / 255.0

X_test = test.loc[:,'pixel0':].values / 255.0

y = train_label
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.1)
X_train.shape
#input reshape

input_shape = (-1,28,28,1)

X_train = X_train.reshape(input_shape)

X_val = X_val.reshape(input_shape)
X_test = X_test.reshape(-1,28,28,1)

# result = model.predict_classes(X_test)
#Now let us encode our labels

y_train = to_categorical(y_train)

y_val = to_categorical(y_val)
#Now we have categoricaly encoded our labels

print(y_train.shape)
def DL_Model(filter1_size=64, filter2_size=32 , activation='relu', optimizer='Adam', padding='Same'):

  model = tf.keras.models.Sequential()

  model.add(tf.keras.layers.Conv2D(filter1_size,(3,3),padding = padding,activation = activation,

                                  input_shape = (28,28,1)))

  model.add(tf.keras.layers.Conv2D(filter1_size,(3,3),padding = padding,activation=activation))

  model.add(tf.keras.layers.Dropout(0.2))

  model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

  model.add(tf.keras.layers.Conv2D(filter2_size,(3,3),padding = padding,activation=activation))

  model.add(tf.keras.layers.Conv2D(filter2_size,(3,3),padding = padding,activation=activation))

  model.add(tf.keras.layers.Dropout(0.25))

  model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

  model.add(tf.keras.layers.Flatten())

  model.add(tf.keras.layers.Dense(128,activation=activation))

  model.add(tf.keras.layers.Dropout(0.25))

  model.add(tf.keras.layers.Dense(256,activation=activation))

  model.add(tf.keras.layers.Dropout(0.25))

  model.add(tf.keras.layers.Dense(10,activation='softmax'))

  model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

  return model
# activation = ['relu','softmax', 'tanh', 'sigmoid', 'linear']

# padding = ['Valid','Same']

# optimizer = ['Adam', 'SGD',  'Adamax','RMSprop']



filter1_size = [64]

filter2_size = [64]

activation = ['relu','tanh','sigmoid']

padding = ['Same']

optimizer = ['Adam','RMSprop']
# param_grid = dict(filter1_size = filter1_size , filter2_size = filter2_size , activation = activation, padding = padding, optimizer = optimizer)



# clf = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn= DL_Model, epochs= 10, batch_size=128, verbose= 3)

# model = GridSearchCV(estimator= clf, param_grid=param_grid, n_jobs=-1, verbose=3)



# epochs = 10

# batch_size = 128

# model.fit(X_train,y_train,

#           validation_data=(X_val,y_val),

#           epochs=epochs,

#           batch_size=batch_size)



# print("Max Accuracy Registred: {} using {}".format(round(model.best_score_,3), 

#                                                    model.best_params_))
model=DL_Model(64,64,'relu','Adam','Same')
epochs = 40

batch_size = 128

model.fit(X_train,y_train,

          validation_data=(X_val,y_val),

          epochs=epochs,

          batch_size=batch_size)
#lets just evaluate the model

model.evaluate(X_val,y_val)
model=DL_Model(64,64,'relu','Adam','Same')
early_stopping_monitor = tf.keras.callbacks.EarlyStopping(

    monitor='val_loss',

    min_delta=0,

    patience=6,

    verbose=0,

    mode='auto',

    baseline=None,

    restore_best_weights=True

)
#This function reduces the learning rate as the training advances whenever validation accuracy decrease.

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
epochs = 30 

batch_size = 86
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
X_train.shape
history = model.fit(datagen.flow(X_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
model.evaluate(X_val,y_val)
%matplotlib inline

def PlotLoss(his, epoch):

    plt.style.use("ggplot")

    plt.figure()

    plt.plot(np.arange(0, epoch), his.history["loss"], label="train_loss")

    plt.plot(np.arange(0, epoch), his.history["val_loss"], label="val_loss")

    plt.title("Training Loss")

    plt.xlabel("Epoch #")

    plt.ylabel("Loss")

    plt.legend(loc="upper right")

    plt.show()



def PlotAcc(his, epoch):

    plt.style.use("ggplot")

    plt.figure()

    plt.plot(np.arange(0, epoch), his.history["accuracy"], label="train_acc")

    plt.plot(np.arange(0, epoch), his.history["val_accuracy"], label="val_accuracy")

    plt.title("Training Accuracy")

    plt.xlabel("Epoch #")

    plt.ylabel("Accuracy")

    plt.legend(loc="upper right")

    plt.show()
PlotLoss(history, 30)

PlotAcc(history, 30)
from sklearn.metrics import confusion_matrix

y_val_predicted = model.predict_classes(X_val)

y_val_actual=np.argmax(y_val, axis=-1)

cm = confusion_matrix(y_val_actual, y_val_predicted)
f, ax = plt.subplots(figsize=(10,10))

sns.heatmap(cm,fmt=".0f", annot=True,linewidths=0.1, linecolor="purple", ax=ax)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()
result = model.predict_classes(X_test)
result
sub_df = test[['id']]

sub_df['label'] = result
sub_df.to_csv('submission.csv',index=False)