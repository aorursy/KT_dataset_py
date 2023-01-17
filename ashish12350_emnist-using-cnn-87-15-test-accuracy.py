#importing the tensorflow
!pip install tensorflow==1.15.3 --user
#importing the numpy and tensorflow
import numpy as np
import tensorflow as tf
train_ds='../input/109-1-ntut-dl-app-hw1/emnist-byclass-train.npz'
test_ds='../input/109-1-ntut-dl-app-hw1/emnist-byclass-test.npz'

df=np.load(train_ds) #loading to the train set

ti=df['training_images'] #ti->train images, calling the label column
df=np.load(train_ds) #loading to the train set
tl=df['training_labels'] #tl->train labels, calling the  label column
# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# instantiating the model in the strategy scope creates the model on the TPU
with tpu_strategy.scope():
    # define your model normally
    model= tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(5, 5),activation='relu'))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Flatten())
    
    model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.15))
    model.add(tf.keras.layers.Dense(62, activation=tf.nn.softmax))

model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(ti, tl, epochs=9,batch_size=128)    

ti = np.load(test_ds)['testing_images']
r=model.predict_classes(ti)
        
# Print results in CSV format and upload to Kaggle
with open('Emnist_pred_results.csv', 'w') as f:
    f.write('Id,Category\n')
    for i in range(len(r)):
        f.write(str(i) + ',' + str(r[i]) + '\n')
# Download your results!
from IPython.display import FileLink
FileLink('Emnist_pred_results.csv')
