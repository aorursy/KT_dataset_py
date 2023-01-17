from __future__ import absolute_import, division, print_function

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import cv2

import matplotlib.pyplot as plt

import seaborn as sns

from PIL import Image

import os

print(os.listdir("../input/cell_images/cell_images"))
infected = os.listdir('../input/cell_images/cell_images/Parasitized/') 

uninfected = os.listdir('../input/cell_images/cell_images/Uninfected/')
data = []

labels = []



for i in infected:

    try:

    

        image = cv2.imread("../input/cell_images/cell_images/Parasitized/"+i)

        image_array = Image.fromarray(image , 'RGB')

        resize_img = image_array.resize((50 , 50))

        rotated45 = resize_img.rotate(45)

        rotated75 = resize_img.rotate(75)

        blur = cv2.blur(np.array(resize_img) ,(10,10))

        data.append(np.array(resize_img))

        data.append(np.array(rotated45))

        data.append(np.array(rotated75))

        data.append(np.array(blur))

        labels.append(1)

        labels.append(1)

        labels.append(1)

        labels.append(1)

        

    except AttributeError:

        print('')

    

for u in uninfected:

    try:

        

        image = cv2.imread("../input/cell_images/cell_images/Uninfected/"+u)

        image_array = Image.fromarray(image , 'RGB')

        resize_img = image_array.resize((50 , 50))

        rotated45 = resize_img.rotate(45)

        rotated75 = resize_img.rotate(75)

        data.append(np.array(resize_img))

        data.append(np.array(rotated45))

        data.append(np.array(rotated75))

        labels.append(0)

        labels.append(0)

        labels.append(0)

        

    except AttributeError:

        print('')
cells = np.array(data)

labels = np.array(labels)



np.save('Cells' , cells)

np.save('Labels' , labels)
print('Cells : {} | labels : {}'.format(cells.shape , labels.shape))
plt.figure(1 , figsize = (15 , 9))

n = 0 

for i in range(49):

    n += 1 

    r = np.random.randint(0 , cells.shape[0] , 1)

    plt.subplot(7 , 7 , n)

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    plt.imshow(cells[r[0]])

    plt.title('{} : {}'.format('Infected' if labels[r[0]] == 1 else 'Unifected' ,

                               labels[r[0]]) )

    plt.xticks([]) , plt.yticks([])

    

plt.show()
plt.figure(1, figsize = (15 , 7))

plt.subplot(1 , 2 , 1)

plt.imshow(cells[0])

plt.title('Infected Cell')

plt.xticks([]) , plt.yticks([])



plt.subplot(1 , 2 , 2)

plt.imshow(cells[60000])

plt.title('Uninfected Cell')

plt.xticks([]) , plt.yticks([])



plt.show()
n = np.arange(cells.shape[0])

np.random.shuffle(n)

cells = cells[n]

labels = labels[n]
cells = cells.astype(np.float32)

labels = labels.astype(np.int32)

cells = cells/255
from sklearn.model_selection import train_test_split



train_x , x , train_y , y = train_test_split(cells , labels , 

                                            test_size = 0.2 ,

                                            random_state = 111)



eval_x , test_x , eval_y , test_y = train_test_split(x , y , 

                                                    test_size = 0.5 , 

                                                    random_state = 111)
plt.figure(1 , figsize = (15 ,5))

n = 0 

for z , j in zip([train_y , eval_y , test_y] , ['train labels','eval labels','test labels']):

    n += 1

    plt.subplot(1 , 3  , n)

    sns.countplot(x = z )

    plt.title(j)

plt.show()
print('train data shape {} ,eval data shape {} , test data shape {}'.format(train_x.shape,

                                                                           eval_x.shape ,

                                                                           test_x.shape))
tf.reset_default_graph()

def cnn_model_fn(features , labels , mode):

    input_layers = tf.reshape(features['x'] , [-1 , 50 , 50 ,3])    

    conv1 = tf.layers.conv2d(

        inputs = input_layers , 

        filters = 50 , 

        kernel_size = [7 , 7],

        padding = 'same',

        activation = tf.nn.relu

        ) 

    

    

    conv2 = tf.layers.conv2d(

        inputs = conv1, 

        filters = 90,

        kernel_size = [3 , 3],

        padding = 'valid',

        activation = tf.nn.relu

        )



        

    conv3 = tf.layers.conv2d(

        inputs = conv2 ,

        filters = 10,

        kernel_size = [5 , 5],

        padding = 'same', 

        activation = tf.nn.relu

        )

    

    pool1 = tf.layers.max_pooling2d(inputs = conv3 , pool_size = [2 , 2] ,

                                    strides = 2 )

    conv4 = tf.layers.conv2d(

        inputs = pool1 ,

        filters = 5,

        kernel_size = [3 , 3],

        padding = 'same', 

        activation = tf.nn.relu

        )



    pool2 = tf.layers.max_pooling2d(inputs = conv4 , pool_size = [2 , 2] ,

                                    strides = 2 , padding = 'same')

    

    pool2_flatten = tf.layers.flatten(pool2)

    fc1 = tf.layers.dense(

        inputs = pool2_flatten,

        units = 2000,

        activation = tf.nn.relu

        )

    fc2 = tf.layers.dense(

        inputs = fc1,

        units = 1000,

        activation = tf.nn.relu

        )

    fc3 = tf.layers.dense(

        inputs = fc2 , 

        units = 500 ,

        activation = tf.nn.relu

        )

    logits = tf.layers.dense(

        inputs = fc3 ,

        units = 2

        )

    

    predictions = {

        'classes': tf.argmax(input = logits , axis = 1),

        'probabilities': tf.nn.softmax(logits , name = 'softmax_tensor')

    }

    

    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(mode = mode , 

                                          predictions = predictions)

    

    loss = tf.losses.sparse_softmax_cross_entropy(labels = labels , 

                                                 logits = logits)

    

    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)

        train_op = optimizer.minimize(loss = loss , 

                                      global_step = tf.train.get_global_step())

        

        return tf.estimator.EstimatorSpec(mode = mode , 

                                            loss = loss , 

                                            train_op = train_op 

                                           )

    eval_metric_op = {'accuracy' : tf.metrics.accuracy(labels = labels ,

                                         predictions =  predictions['classes'])}

    

    return tf.estimator.EstimatorSpec(mode = mode , 

                                      loss = loss , 

                                      eval_metric_ops = eval_metric_op)    
malaria_detector = tf.estimator.Estimator(model_fn = cnn_model_fn , 

                                         model_dir = '/tmp/modelchkpt')
tensors_to_log = {'probabilities':'softmax_tensor'}

logging_hook = tf.train.LoggingTensorHook(

    tensors = tensors_to_log , every_n_iter = 50 

    )
train_input_fn = tf.estimator.inputs.numpy_input_fn(

    x = {'x': train_x},

    y = train_y,

    batch_size = 100 , 

    num_epochs = None , 

    shuffle = True

    )

malaria_detector.train(input_fn = train_input_fn , steps = 1 , hooks = [logging_hook])
malaria_detector.train(input_fn = train_input_fn , steps = 10000)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(

    x = {'x': eval_x},

    y = eval_y , 

    num_epochs = 1 , 

    shuffle = False

    )

eval_results = malaria_detector.evaluate(input_fn = eval_input_fn)

print(eval_results)
pred_input_fn = tf.estimator.inputs.numpy_input_fn(

    x = {'x' : test_x},

    y = test_y,

    num_epochs = 1,

    shuffle = False

    )



y_pred = malaria_detector.predict(input_fn = pred_input_fn)

classes = [p['classes'] for p in y_pred]
from sklearn.metrics import confusion_matrix , classification_report , accuracy_score

print('{} \n{} \n{}'.format(confusion_matrix(test_y , classes) , 

                           classification_report(test_y , classes) , 

                           accuracy_score(test_y , classes)))
plt.figure(1 , figsize = (15 , 9))

n = 0 

for i in range(49):

    n += 1 

    r = np.random.randint( 0  , test_x.shape[0] , 1)

    plt.subplot(7 , 7 , n)

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    plt.imshow(test_x[r[0]])

    plt.title('true {} : pred {}'.format(test_y[r[0]] , classes[r[0]]) )

    plt.xticks([]) , plt.yticks([])



plt.show()