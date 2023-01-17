import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np

import os
def reset_graph(seed=42): ### it resests all created graph, it's required once re-defining of any placeholders, variables, shapes or model structures is needed

    tf.reset_default_graph()

    tf.set_random_seed(seed)

    np.random.seed(seed)
def generate_batch(x_train, y_train, batch_size): ### it takes random permutation of lenght x_train and splits x_train (together with y_train) into batches number

    rnd_idx = np.random.permutation(len(x_train))

    n_batches = len(x_train) // batch_size

    for batch_idx in np.array_split(rnd_idx, n_batches):

        x_batch, y_batch = x_train[batch_idx], y_train[batch_idx]

        yield x_batch, y_batch
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() ### loading the datasets

x_train = x_train.astype(np.float32).reshape(-1, 28*28) / 255.0 ### reshaping and normalizing

x_test = x_test.astype(np.float32).reshape(-1, 28*28) / 255.0 ### reshaping and normalizing

y_train = y_train.astype(np.int32)

y_test = y_test.astype(np.int32)
print("Length of the training set:", len(x_train))

print("Length of the test set:", len(x_test))
n_inputs = 28 ### lengh of each row

n_steps = 28 ### number of time steps
reset_graph()
with tf.name_scope("Inputs"):

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="X")

    y = tf.placeholder(tf.int32, [None], name="y")

    keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_probability')

print(X)

print(y)

print(keep_prob)
### In the code:

#with tf.Session() as sess:

    #sess.run(tf.global_variables_initializer())

    #writer = tf.summary.FileWriter("path of the project", sess.graph)

    

### In the shell:

#--tensorboard --logdir="path_to_filewirter" --port 6006
n_layers = 3 ### number of BasicRNNCell layers

n_neurons = 100 ### number of neurons in the network

n_outputs = 10 ### outputs that represent digits from 0-9
with tf.name_scope("Basic_RNN_Layers"):

    lstm_cells = [tf.nn.rnn_cell.BasicRNNCell(num_units = n_neurons, activation = tf.nn.relu)

             for layer in range(n_layers)]

    lstm_cells_drop = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)

                for cell in lstm_cells]

    multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells_drop)

    outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype = tf.float32) ### states return final state (last output) of the multi_layer_cell
with tf.name_scope("Loss"):

    states_concat = tf.concat(axis=1, values=states, name='states_reshape')

    dense1 = tf.layers.dense(states_concat, 64, name='dense_1')

    dense2 = tf.layers.dense(dense1, 32, name='dense_2')

    logits = tf.layers.dense(dense2, n_outputs, name='output_layer')

    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=tf.reshape(logits, shape=(-1, n_outputs)), name='softmax_cross_entropy')

    loss = tf.reduce_mean(xentropy, name='loss')

    loss_summary = tf.summary.scalar('loss_summ', loss)
with tf.name_scope("Train"):    

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='Adam_optimizer')

    training_optimizer = optimizer.minimize(loss, name='training_Adam')
with tf.name_scope("Evaluation"):        

    correct = tf.nn.in_top_k(tf.reshape(logits, (-1, n_outputs)), y, 1, name='inTopK')

    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='Accuracy')

    accuracy_summary = tf.summary.scalar('Accuracy_Summ', accuracy)
init = tf.global_variables_initializer()

saver = tf.train.Saver()
train_keep_prob = 0.8
x_test = x_test.reshape((-1, n_steps, n_inputs)) ### reshaping test set
from datetime import datetime



def log_dir(prefix=""):

    now = datetime.utcnow().strftime('%Y-%m-%d-%H-%m-%S')

    root_logdir = "TensorFlow_Logs"

    if prefix:

        prefix += '-'

    name = prefix + now

    return '{}/{}/'.format(root_logdir, name)
logdir=log_dir("mnist_rnn_model")
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
checkpoint_path = "/checkpoints/mnist_rnn_model.ckpt"

checkpoint_epoch_path = checkpoint_path + ".epoch"

final_model_path = "./mnist_rnn_model"
best_loss = np.infty                ### parameters for early stopping

epochs_without_progress = 0         ### once epochs_without_progress reaches the value

max_epochs_without_progress = 15    ### of max_epochs_without_progress, the model stops and saves last parameters
acc_list, acc_test_list, loss_list, loss_test_list = [], [], [], [] 
n_epochs = 300

batch_size = 150



with tf.Session() as sess:

    init.run() 

    for epoch in range(n_epochs):

        for x_batch, y_batch in generate_batch(x_train, y_train, batch_size): ### generating batches for x/y_train

            x_batch = x_batch.reshape((-1, n_steps, n_inputs)) ### reshape to the format define for X placeholder

            

            ### for x_batch/y_batch data we feed keep_prob values for 0.8

            sess.run(training_optimizer, feed_dict={X: x_batch, y: y_batch, keep_prob: train_keep_prob})

        acc_batch, loss_batch, acc_sum, loss_sum = sess.run([accuracy, loss, accuracy_summary, loss_summary], feed_dict={X: x_batch, y: y_batch, keep_prob: train_keep_prob})   

        

        ### during evaluation on test set, no dropout required, keep_prob has default value (1.0)    

        acc_test, loss_test, acc_test_sum, loss_test_sum = sess.run([accuracy, loss, accuracy_summary, loss_summary], feed_dict={X: x_test, y: y_test})

               

        acc_list.append(acc_batch)        #### if not using tensorboard,

        loss_list.append(loss_batch)      #### temporary parameters of train/test accuracy and loss 

        acc_test_list.append(acc_test)    #### can be placed in lists and plotted accordingly

        loss_test_list.append(loss_test)  

            

        file_writer.add_summary(acc_sum, epoch)

        file_writer.add_summary(loss_sum, epoch)

        file_writer.add_summary(acc_test_sum, epoch)

        file_writer.add_summary(loss_test_sum, epoch)

        

        if epoch % 5 == 0:

            print("Epoch", epoch,

                  '\tValidation accuracy: {:.3f}%'.format(acc_batch * 100), '\tTest accuracy: {:.3f}%'.format(acc_test * 100), '\tLoss: {:.3f}'.format(loss_batch))

            saver.save(sess, checkpoint_path)

            with open(checkpoint_epoch_path, "wb") as f:

                f.write(b'%d' % (epoch + 1))

            if loss_batch < best_loss:

                saver.save(sess, final_model_path)

                best_loss = loss_batch

            else:

                epochs_without_progress += 2

                if epochs_without_progress > max_epochs_without_progress:

                    print('Early Stopping')

                    break
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))

fig.suptitle('Evaluation results')

ax1.set_title("Train Accuracy vs Test Accuracy")

ax1.plot(range(len(acc_test_list)), acc_list, color="#cc802f", label="Train Acc")

ax1.plot(range(len(acc_test_list)), acc_test_list, color="#1649a8", label="Test Acc")

ax1.legend(loc='lower right')

ax2.set_title("Train Loss vs Test Loss")

ax2.plot(range(len(acc_test_list)), loss_list, color="#cc802f", label="Train Loss")

ax2.plot(range(len(acc_test_list)), loss_test_list, color="#1649a8", label="Test Loss")

ax2.legend(loc='best')
with tf.Session() as sess:

    saver.restore(sess, final_model_path)

    probabilities = logits.eval(feed_dict={X: x_test, y: y_test})
probabilities.shape ### the shape contains (number of records, number of outputs)
ax_s = []

for i in range(8):

    ax_s.append(str("ax")+ str(i))





def plot_preditions():

    fig, (ax_s) = plt.subplots(1, 8, figsize=(22,6))



    perm = np.random.permutation(len(x_test))

    perm = perm[:8]

    for i in range(8):

        title = "Predicted value:" + str(np.argmax(probabilities[perm[i]])) + ",\nTrue value: " + str(y_test[perm[i]])

        ax_s[i].set_title(title)

        ax_s[i].imshow(x_test[perm[i]],cmap =plt.cm.gray_r, interpolation = "nearest")    
plot_preditions()

plot_preditions()

plot_preditions()

plot_preditions()