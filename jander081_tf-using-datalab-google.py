import pandas as pd, numpy as np, tensorflow as tf, sys

import matplotlib.pyplot as plt, matplotlib.cm as cm
%matplotlib inline

import warnings
warnings.filterwarnings(action='ignore')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

import google.datalab.storage as storage
from io import BytesIO
# ONCE YOUR DATA IS UPLOADED INTO A BUCKET, THE PATH CAN BE FOUND 
# UNDER THE OVERVIEW TAB
%%gcs list --object gs://pathtobucket
# this will return a list of a given bucket's contents
# MUST BE RUN IN A SEPARATE CHUNK
%gcs read --object "gs://digits_081/test.csv" --variable data_test
# the variable assignment is whatever you like
# CONGRATS! YOU NOW HAVE YOUR DATAFRAME
X_test = pd.read_csv(BytesIO(data_test))
# TRUNCATED THE DATASETS IN ORDER TO RUN ON CPU
X = pd.read_csv('../input/train.csv').iloc[:5000, :] 
X_test = pd.read_csv('../input/test.csv').iloc[:500, :] 

labels = X.iloc[:, 0]
X.drop(['label'], axis=1, inplace=True)

print(X.shape); print(X_test.shape); print(labels.shape); X.head(2)
image_size = X.shape[1]
print("Number of pixels per image: {} ranging {} to {}".format(image_size, X.values.min(), X.values.max()))

image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
print("Image size: {}x{}".format(image_width, image_height))

num_colors = 1 # black and white
for i in range(0, 9):
    image = X.iloc[i, :].values
    plt.subplot(3, 3, i+1)
    plt.imshow(image.reshape(image_width, image_height), cmap=cm.binary)
    plt.axis('off')

plt.tight_layout()
plt.show()
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, labels,
                                                      test_size=0.1, 
                                                      random_state=81)
dframes = [X_train, X_valid, y_train, y_valid]

for frame in dframes:
    frame.reset_index(drop=True, inplace=True)
    print(frame.shape)
# CONVERT TO ARRAY AND RESHAPE
X_train = X_train.values.reshape([-1, image_width, image_height, 1]).astype("float32")
X_valid = X_valid.values.reshape([-1, image_width, image_height, 1]).astype("float32")
X_test = X_test.values.reshape([-1, image_width, image_height, 1]).astype("float32")
print('{} \n{}'.format(X_train.shape, X_valid.shape))
def genAugmentedImg(img, label, num_data=10000):
    gen_data = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest')
    
    gen_data.fit(X_train)
    data = gen_data.flow(img, label, batch_size=num_data)
    
    return data[0][0], data[0][1]
X_aug_train, y_aug_train = genAugmentedImg(X_train, y_train, 200) # used 20K for model
X_aug_valid, y_aug_valid = genAugmentedImg(X_valid, y_valid, 50) # used 5K

print("X_aug_train shape :", X_aug_train.shape)
print("y_aug_train shape :", y_aug_train.shape)

# JUST TAKING A LOOK - CALL ME CURIOUS
for i in range(0, 9):
    image = X_aug_train[i, :, :, 0]
    plt.subplot(3, 3, i+1)
    plt.imshow(image, cmap=cm.binary)
    plt.axis('off')
plt.tight_layout()
plt.show()

# YOU DON'T WANT THE AUGMENTED IMAGES USED FOR VALIDATION, SO JUST
# LUMP THEM BACK IN WITH THE TRAINING SET
X_train = np.concatenate((X_train, X_aug_train, X_aug_valid), axis=0)
y_train = np.concatenate((y_train, y_aug_train, y_aug_valid), axis=0)
# TOTALLY UNNECESSARY, COULD JUST DO X_train = X_train/255.0
def normalizer(dframe): 
    dframe = (dframe - X_train.min()) / (X_train.max() - X_train.min())
    return(dframe)

X_train = normalizer(X_train)
X_valid = normalizer(X_valid)
X_test = normalizer(X_test)
from keras.utils.np_utils import to_categorical 

y_train = to_categorical(y_train, num_classes = 10)
y_valid= to_categorical(y_valid, num_classes = 10)

num_labels = y_train.shape[1]

# RAN A MINI TEST TO MAKE SURE THIS WORKS
def shuffleData(x, y): 
    idx = np.arange(0, x.shape[0])
    np.random.shuffle(idx)    
    return x[idx], y[idx]
# NOT BAD FOR A NEXT_BATCH FUNCTION. THIS DOES NOT ACCOUNT FOR REMAINDERS 
# AT THE TAIL END OF AN EPOCH. BUT THE SHUFFLE FUNCTION NEGATES 
# THE NEED FOR IT. REMAINDER OR NOT, THE IMAGES ARE RESHUFFLED 
# BETWEEN EPOCHS. SO EVERY IMAGE WILL BE USED.
epochs_completed = 0
index_in_epoch = 0

def next_batch(x, y, batch_size):

    global index_in_epoch
    global epochs_completed
       
    num_examples = x.shape[0]
    start = index_in_epoch
    index_in_epoch += batch_size

    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        x, y = shuffleData(x, y)
        # start next epoch
        start = 0
        index_in_epoch = batch_size  
    end = index_in_epoch
    # we don't need to store past batches in memory 
    yield x[start:end], y[start:end]
tf.reset_default_graph()
tf.set_random_seed(81)
np.random.seed(81)

alpha = 0.1
x_kernel = tf.contrib.layers.xavier_initializer()


# PLACEHOLDERS
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, num_labels])
learning_rate = tf.placeholder(tf.float32)
non_dropout = tf.placeholder(tf.float32)

# INPUT
images = tf.reshape(x, [-1, image_width, image_height, num_colors])
# (batch, height, width, channel)

# CNN LAYERS
conv1_a = tf.layers.conv2d(inputs=images, filters=32, kernel_size=5, strides=(1,1),
                        padding='same',activation=tf.nn.relu,
                        kernel_initializer=x_kernel)  
conv1_b = tf.layers.conv2d(inputs=conv1_a, filters=32, kernel_size=5, strides=(1,1),
                        padding='same',activation=tf.nn.relu,
                        kernel_initializer=x_kernel)  
pool1 = tf.layers.max_pooling2d(conv1_b,pool_size=2, strides=2)
drop1 = tf.nn.dropout(pool1, keep_prob=0.75)

conv2_a = tf.layers.conv2d(drop1, 64, 5, padding='same', activation=tf.nn.relu, 
                           kernel_initializer=x_kernel)
conv2_b = tf.layers.conv2d(conv2_a, 64, 5, padding='same', activation=tf.nn.relu, 
                           kernel_initializer=x_kernel)
pool2 = tf.layers.max_pooling2d(conv2_b,pool_size=2, strides=2)
drop2 = tf.nn.dropout(pool2, keep_prob=0.75)


# FULL CONNECTED LAYERS
fc1_a = tf.contrib.layers.flatten(drop2)
fc1_b = tf.layers.dense(fc1_a, 256, kernel_initializer=x_kernel)
fc1_c = tf.layers.batch_normalization(fc1_b)
fc1_d = tf.maximum(fc1_c, fc1_c*alpha)
fc1_e = tf.nn.dropout(fc1_d, non_dropout)

fc2_a = tf.contrib.layers.flatten(fc1_e)
fc2_b = tf.layers.dense(fc2_a, 128, kernel_initializer=x_kernel)
fc2_c = tf.layers.batch_normalization(fc2_b)
fc2_d = tf.maximum(fc2_c, fc2_c*alpha)
fc2_e = tf.nn.dropout(fc2_d, non_dropout)

logits = tf.layers.dense(fc2_e, 10, kernel_initializer=x_kernel)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

pred = tf.argmax(logits, 1)

# chart
chart = {}
chart["train_acc"] = []
chart["val_acc"] = []

def train(epochs, batch_size, keep_probability, lr):
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(epochs + 1):

            for x_batch, y_batch in next_batch(X_train, y_train, batch_size):  
                sess.run(optimizer, feed_dict={x: x_batch, y: y_batch, non_dropout: keep_probability, learning_rate:lr})

            accuracy_train = accuracy.eval({x: x_batch, y: y_batch, non_dropout: 1.0})
            accuracy_val = accuracy.eval({x: X_valid, y: y_valid, non_dropout: 1.0})

            # PRINT OUT A MESSAGE EVERY 100 STEPS
            if i%100 == 0:
                print('epoch {}'.format(i))
                print('Training acc: {:.4f} \n Validation acc: {:.4f}'.format(accuracy_train, accuracy_val))
                
                chart['train_acc'].append(accuracy_train)
                chart['val_acc'].append(accuracy_val)
                
        output = pred.eval({x: X_test, non_dropout: 1.0})
            
    return output

# %%time
# WILL RUN A SHORTENED VERSION WITH CPU 
epochs = 500 # This was 10000 with abovemention GPU, about 40min 
batch_size = 128
keep_probability = 0.50
lr = 0.001

predictions = train(epochs, batch_size, keep_probability, lr)
df_out = pd.DataFrame(predictions)
df_out.index = [x + 1 for x in df_out.index]
df_out.rename(columns={0: 'Label'}, inplace=True)
df_out.index.name = "ImageId"

df_out.to_csv("predictions_4.csv", index=True)
df_out.head()
# TAKE A LOOK AT THE FIRST FEW. IF YOU GOT THE FIRST 5 CORRECT, 
# YOU'RE LOOKING GOOD. IF YOU MISSED THE 2ND ZERO ONLY, YOU'RE
# STILL DOING OK. ANY MORE THAN THAT, BOO
for index in range(5):
    plt.figure()
    image = X_test[index,:]
    plt.imshow(image.reshape(image_width, image_height), cmap=cm.binary)
# NOT REALLY HELPFUL WITH THE REDUCED DATASET
# plot acc
plt.plot(chart["train_acc"])
plt.plot(chart["val_acc"])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(['train', 'validation'], loc=4),
plt.show()
!gsutil cp 'predictions.csv' 'gs://path/predictions.csv'
# NOW RUN THE FOLLOWING TO CONFIRM YOUR CSV HAS BEEN ADDED TO 
# YOUR BUCKET
%%gcs list --object gs://pathtobucket