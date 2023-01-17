import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

from scipy import stats

from scipy.stats import norm, skew

from sklearn.model_selection import train_test_split
train_df_raw = pd.read_csv('../input/train.csv')

y_train = train_df_raw.label

train_df_raw.drop('label', inplace = True, axis = 1)

x_train = train_df_raw

x_test = pd.read_csv('../input/test.csv')

x = pd.concat([x_train, x_test], ignore_index = True)

x_train.shape, x_test.shape, x.shape
x.describe()
x = x/255

x_raw = x
from sklearn.decomposition import PCA

pca = PCA(n_components = 200)

x = pca.fit_transform(x_raw)

x_train = pd.DataFrame(x[:42000])

x_test = pd.DataFrame(x[42000:])
x_train.shape, x_test.shape
def scores_cv(model):

    kf = KFold(5, shuffle = True, random_state = 0).get_n_splits(x_train.values)

    return np.mean(cross_val_score(model, x_train.values, y_train.values, scoring = "accuracy",cv = kf, n_jobs = -1))



def scores_train_test(model):

    x_m_train, x_m_test, y_m_train, y_m_test = train_test_split(x_train.values, y_train.values, test_size=0.3, random_state=42)

    model.fit(x_m_train, y_m_train)

    return (model.score(x_m_train, y_m_train), model.score(x_m_test, y_m_test))
from sklearn.linear_model import  LogisticRegression

from sklearn.model_selection import KFold, train_test_split, cross_val_score

lin_reg_model = LogisticRegression(random_state=0,

                         multi_class='multinomial',

                          solver='saga', tol=0.1)

scores_cv(lin_reg_model)
scores_train_test(lin_reg_model)
from sklearn.svm import LinearSVC

svm_model= LinearSVC(random_state = 0, tol = 0.1, max_iter = 1000)

scores_cv(svm_model)
scores_train_test(svm_model)
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors = 5, weights = 'distance', n_jobs = -1)

scores_cv(knn_model)

    
scores_train_test(knn_model)
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier,RandomForestClassifier
extratrees_model = ExtraTreesClassifier(n_estimators = 500, min_samples_leaf = 20, bootstrap = True, oob_score = True, n_jobs = -1)

scores_cv(extratrees_model)
scores_train_test(extratrees_model)
rf_model = RandomForestClassifier(n_estimators = 200, min_samples_leaf = 20, bootstrap = True, oob_score = True, n_jobs = -1, max_features = 'sqrt')

scores_cv(rf_model)
scores_train_test(rf_model)
gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, tol = 0.005,

                                   loss='deviance', random_state =5)

scores_cv(gb_model)
scores_train_test(gb_model)
from sklearn.svm import SVC

svm_non_linear_model= SVC(random_state = 0, tol = 0.1, max_iter = 1000)

scores_cv(svm_non_linear_model)
scores_train_test(svm_non_linear_model)
import tensorflow as tf

x = x_raw

x_train = x[:42000]

x_test = x[42000:]

x_train.shape, x_test.shape
fc_nn_model = tf.keras.models.Sequential([

  tf.keras.layers.Dense(512, activation=tf.nn.relu),

    tf.keras.layers.Dense(128, activation = tf.nn.relu),

  tf.keras.layers.Dense(64, activation = tf.nn.relu),

  tf.keras.layers.Dense(10, activation=tf.nn.softmax)

])

fc_nn_model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])

x_m_train, x_m_test, y_m_train, y_m_test = train_test_split(x_train.values, y_train.values, test_size=0.2, random_state=42)

fc_nn_model.fit(x_m_train, y_m_train, epochs = 25, batch_size = 512, steps_per_epoch = 8)

fc_nn_model.evaluate(x_m_train, y_m_train), fc_nn_model.evaluate(x_m_test, y_m_test)
from keras.preprocessing.image import ImageDataGenerator

image_generator = ImageDataGenerator(rotation_range=10,

            zoom_range = 0.05, 

            width_shift_range=0.05,

            height_shift_range=0.05,

            horizontal_flip=False,

            vertical_flip=False, 

            data_format="channels_last",

            )

x_m_train, x_m_test, y_m_train, y_m_test = train_test_split(x_train.values, y_train.values, test_size=0.2, random_state=42)

x_m_train = x_m_train.reshape((-1,28,28,1))

x_m_test = x_m_test.reshape((-1,28,28,1))

cnn_model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(8, (3,3), activation = 'relu', padding = 'same', input_shape = (28, 28, 1)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(16, (5,5), activation = 'relu', padding = 'valid'),

    tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(120, activation = 'relu'),

    tf.keras.layers.Dense(84, activation = 'relu'),

    tf.keras.layers.Dense(10, activation = 'softmax')

])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cnn_model.fit_generator(image_generator.flow(x_m_train, y_m_train, batch_size=128),

                    steps_per_epoch= 200, epochs=30)

cnn_model.evaluate(x_m_train, y_m_train), cnn_model.evaluate(x_m_test, y_m_test)
from keras.applications.resnet50 import ResNet50

from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Conv2D, Input

from keras import Model

input_tensor = Input(shape=(28,28,1))

base_resnet = ResNet50(weights = None, include_top = False, input_tensor = input_tensor)

x = base_resnet.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.7)(x)

predictions = Dense(10, activation = 'softmax')(x)

resnet_model = Model(inputs = base_resnet.input, outputs = predictions)

resnet_model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
resnet_model.summary()
image_generator = ImageDataGenerator(rotation_range=10,

            zoom_range = 0.05, 

            width_shift_range=0.05,

            height_shift_range=0.05,

            horizontal_flip=False,

            vertical_flip=False, 

            data_format="channels_last",

            )

x_m_train, x_m_test, y_m_train, y_m_test = train_test_split(x_train.values, y_train.values, test_size=0.1, random_state=42)

x_m_train = x_m_train.reshape((-1,28,28,1))

x_m_test = x_m_test.reshape((-1,28,28,1))

resnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

resnet_model.fit_generator(image_generator.flow(x_m_train, y_m_train, batch_size=256), validation_data = (x_m_test,y_m_test),

                    steps_per_epoch= 500, epochs=100)

resnet_model.evaluate(x_m_train, y_m_train), resnet_model.evaluate(x_m_test, y_m_test)
resnet_model.fit_generator(image_generator.flow(x_m_train, y_m_train, batch_size=128),

                    steps_per_epoch= 200, epochs=30)

resnet_model.evaluate(x_m_train, y_m_train), resnet_model.evaluate(x_m_test, y_m_test)
x_m_train = x_train.values

x_m_train = x_m_train.reshape((-1,28,28,1))

y_m_train = y_train.values

resnet_model.fit_generator(image_generator.flow(x_m_train, y_m_train, batch_size=128),steps_per_epoch= 200, epochs=30)

resnet_model.evaluate(x_m_train, y_m_train)
x_m_test = x_test.values.reshape((-1,28,28,1))

predictions = resnet_model.predict(x_m_test)

predictions_df = pd.DataFrame({'ImageId': np.arange(1,28001), 'Label': [np.argmax(prediction) for prediction in predictions]})

predictions_df.to_csv('./outputs.csv')



from IPython.display import HTML

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index = False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create_download_link(predictions_df)
np.argmax(predictions[2])
plt.imshow(x_m_test[2].reshape((28,28)), cmap='gray', vmin=0., vmax=1.)
np.argmax(predictions)
class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):

    if(logs.get('acc')>= 0.9999):

      print("\nReached 99.99% accuracy so cancelling training!")

      self.model.stop_training = True

        

callbacks = myCallback()

mnist = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images=training_images.reshape(60000, 28, 28, 1)

training_images=training_images / 255.0

test_images = test_images.reshape(10000, 28, 28, 1)

test_images=test_images/255.0

training_images = np.concatenate([training_images,test_images])

training_labels = np.concatenate([training_labels, test_labels])

model = tf.keras.models.Sequential([

  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),

  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Flatten(),

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dense(10, activation='softmax')

])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'] )

model.fit(training_images, training_labels, epochs=50, callbacks = [callbacks])

x_m_test = x_test.values.reshape((-1,28,28,1))

predictions = model.predict(x_m_test)

predictions_df = pd.DataFrame({'ImageId': np.arange(1,28001), 'Label': [np.argmax(prediction) for prediction in predictions]})

create_download_link(predictions_df)
x_test.values.shape