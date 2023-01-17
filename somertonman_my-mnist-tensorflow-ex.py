!pip install tensorflow_datasets

!pip install tensorflow==2.0.0rc1

import numpy as np

import tensorflow as tf

import tensorflow_datasets as tfds

print(tf.__version__)
mnist_dataset,mnist_info = tfds.load(name='mnist',as_supervised=True,with_info=True)



mnist_train,mnist_test = mnist_dataset['train'],mnist_dataset['test']



number_val_samples = 0.1 * mnist_info.splits['train'].num_examples

number_val_samples = tf.cast(number_val_samples, tf.int64)



number_test_samples = mnist_info.splits['test'].num_examples

number_test_samples = tf.cast(number_test_samples, tf.int64)
def scale(image,label):

    image = tf.cast(image, tf.float32)

    image /= 255.

    return image,label



scaled_train_val_data = mnist_train.map(scale)

scaled_test_data = mnist_test.map(scale)

# size =1 means not shuffling at all, number of objects to be put in buffer at each iter



buffer_size=10000



shuffled_scaled_train_val_data = scaled_train_val_data.shuffle(buffer_size)
validation_data = shuffled_scaled_train_val_data.take(number_val_samples)

train_data = shuffled_scaled_train_val_data.skip(number_val_samples)
#samples to take in each batch



batch_size = 100



train_data = train_data.batch(batch_size)

validation_data = validation_data.batch(number_val_samples)

test_data= scaled_test_data.batch(number_test_samples)



validation_inputs, validation_targets = next(iter(validation_data))
input_size = 784

output_size = 10

hidden_layer_size = 100





model = tf.keras.Sequential ([

    

    # the first layer (the input layer)

    # each observation is 28x28x1 pixels, therefore it is a tensor of rank 3

    # since we don't know CNNs yet, we don't know how to feed such input into our net, so we must flatten the images

    # there is a convenient method 'Flatten' that simply takes our 28x28x1 tensor and orders it into a (None,) 

    # or (28x28x1,) = (784,) vector

    # this allows us to actually create a feed forward neural network

    

    tf.keras.layers.Flatten(input_shape=(28,28,1)),

    

    # tf.keras.layers.Dense is basically implementing: output = activation(dot(input, weight) + bias)

    # it takes several arguments, but the most important ones for us are the hidden_layer_size and the activation function

    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),

    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),

    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),

    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),

    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),

    

    # the final layer is no different, we just make sure to activate it with softmax

    tf.keras.layers.Dense(output_size, activation='softmax')

])



#model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
num_of_epochs = 5

tensorboard_callback = tf.keras.callbacks.TensorBoard("logs")

model.fit(train_data, epochs=num_of_epochs, validation_data=(validation_inputs, validation_targets), verbose =2, validation_steps=1)
import matplotlib.pyplot as plt

#x = test_data.next_batch(1)





#  plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap("gray"))

#  print("Label: %d" % label.numpy())



x = mnist_test.take(10).__iter__()



#np.array(x,dtype='float')

#next(x)



#z = np.array(next(x))



#z.shape



#plt.gray()

#plt.imshow(z)



#tf.reshape(z, [28, 28])
%load_ext tensorboard

%tensorboard --logdir logs