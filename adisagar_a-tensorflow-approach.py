import tensorflow as tf

import pandas as pd

import numpy as np



testing_size = 2000



epochs_completed = 0

index_in_epoch = 0





output = tf.placeholder(tf.float32, [None, 10])

input = tf.placeholder(tf.float32, [None, 784])



data = pd.read_csv("../input/train.csv")



imageset = data.iloc[:, 1:]

labels_flat = data[[0]].values.ravel()





def dense_to_one_hot(labels_dense, num_classes):

    num_labels = labels_dense.shape[0]

    index_offset = np.arange(num_labels) * num_classes

    labels_one_hot = np.zeros((num_labels, num_classes))

    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot





labels = dense_to_one_hot(labels_flat, np.unique(labels_flat).shape[0])



validation_images = imageset[:testing_size]

validation_labels = labels[:testing_size]



train_images = imageset[testing_size:]

train_labels = labels[testing_size:]



num_examples = train_images.shape[0]



# serve data by batches

def next_batch(batch_size):

    

    global train_images

    global train_labels

    global index_in_epoch

    global epochs_completed

    

    start = index_in_epoch

    index_in_epoch += batch_size

    

    # when all trainig data have been already used, it is reorder randomly    

    if index_in_epoch > num_examples:

        # finished epoch

        epochs_completed += 1

        # shuffle the data

        perm = np.arange(num_examples)

        np.random.shuffle(perm)

        train_images = train_images[perm]

        train_labels = train_labels[perm]

        # start next epoch

        start = 0

        index_in_epoch = batch_size

        assert batch_size <= num_examples

    end = index_in_epoch

    return train_images[start:end], train_labels[start:end]





#Define level one -



weights_1_layer = tf.Variable(tf.random_normal([784, 500]))

bias_1_layer = tf.Variable(tf.random_normal([ 500]))

level_1_layer =  tf.nn.relu(tf.matmul(input,weights_1_layer) + bias_1_layer)



weights_output = tf.Variable(tf.random_normal([500, 10]))

bias_output = tf.Variable(tf.random_normal([ 10]))

prediction = tf.matmul(level_1_layer,weights_output) + bias_output

predict_onehot = tf.argmax(prediction,1)







cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output) )

optimizer = tf.train.AdamOptimizer().minimize(cost)



batch_size =100

epochs = 10

totalcost = 0

count , _ = train_images.shape





with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())

	for epoch in range(epochs):

		for _ in range(int(count/batch_size)):

			x, y = next_batch(batch_size)

			_ , c = sess.run([optimizer,cost], feed_dict = {input: x, output: y })

			totalcost += c

		print('Executed Epoch - ', epoch ,'Total cost - ' , totalcost)

		totalcost = 0

		index_in_epoch = 0

		epochs_completed = 0



	correct = tf.equal(tf.argmax(prediction,1), tf.argmax(output,1))

	accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

	print('Accuracy:', accuracy.eval({input:validation_images, output:validation_labels}))



	test_data = pd.read_csv('../input/test.csv')

	predicted_lables = np.zeros(test_data.shape[0])

	

	for i in range(int(test_data.shape[0]/batch_size)):

		x = test_data[i*batch_size : (i+1)*batch_size]

		predicted_lables[i*batch_size : (i+1)*batch_size] = predict_onehot.eval(feed_dict = { input : x})



	np.savetxt('sample_submission.csv', 

	           np.c_[range(1,len(test_data)+1),predicted_lables], 

	           delimiter=',', 

	           header = 'ImageId,Label', 

	           comments = '', 

	           fmt='%d')