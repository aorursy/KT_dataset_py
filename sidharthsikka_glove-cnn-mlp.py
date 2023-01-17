import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

import json

import string

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

import math

import pickle

from tqdm import tqdm



Q6_TRAIN_CLAIMS_ARR = '../input/train_claim_arr.pickle'

Q6_TRAIN_SENT_ARR = '../input/train_sent_arr.pickle'

Q6_TRAIN_LABEL_ARR = '../input/train_label_arr.pickle'

Q6_DEV_CLAIMS_ARR = '../input/dev_claim_arr.pickle'

Q6_DEV_SENT_ARR = '../input/dev_sent_arr.pickle'

Q6_DEV_LABEL_ARR = '../input/dev_label_arr.pickle'

Q6_TEST_CLAIMS_ARR = '../input/test_claim_arr.pickle'

Q6_TEST_SENT_ARR = '../input/test_sent_arr.pickle'

Q6_TEST_LABEL_ARR = '../input/test_label_arr.pickle'

Q6_TRAIN_CLAIMS_ARR_BALANCED = '../input/train_claim_arr_balanced.pickle'

Q6_TRAIN_SENT_ARR_BALANCED = '../input/train_sent_arr_balanced.pickle'

Q6_TRAIN_LABEL_ARR_BALANCED = '../input/train_label_arr_balanced.pickle'

Q6_DEV_CLAIMS_ARR_BALANCED = '../input/dev_claim_arr_balanced.pickle'

Q6_DEV_SENT_ARR_BALANCED = '../input/dev_sent_arr_balanced.pickle'

Q6_DEV_LABEL_ARR_BALANCED = '../input/dev_label_arr_balanced.pickle'

Q6_GLOVE_VOCAB = '../input/q6_vocab_to_index_glove.pickle'

Q6_GLOVE_EMBEDDING = '../input/q6_embedding_glove.pickle'
with open(Q6_TRAIN_CLAIMS_ARR_BALANCED, 'rb') as openFile:

    data_claim = np.array(pickle.load(openFile), dtype=np.uint16)

with open(Q6_TRAIN_SENT_ARR_BALANCED, 'rb') as openFile:

    data_sent = np.array(pickle.load(openFile), dtype=np.uint16)

with open(Q6_TRAIN_LABEL_ARR_BALANCED, 'rb') as openFile:

    data_labels = np.array(pickle.load(openFile), dtype=np.uint8).reshape((len(data_sent), 1))

with open(Q6_DEV_CLAIMS_ARR_BALANCED, 'rb') as openFile:

    data_claim_dev = np.array(pickle.load(openFile), dtype=np.uint16)

with open(Q6_DEV_SENT_ARR_BALANCED, 'rb') as openFile:

    data_sent_dev = np.array(pickle.load(openFile), dtype=np.uint16)

with open(Q6_DEV_LABEL_ARR_BALANCED, 'rb') as openFile:

    data_labels_dev = np.array(pickle.load(openFile), dtype=np.uint8).reshape((len(data_sent_dev), 1))

with open(Q6_TEST_CLAIMS_ARR, 'rb') as openFile:

    data_claim_test = np.array(pickle.load(openFile), dtype=np.uint16)

with open(Q6_TEST_SENT_ARR, 'rb') as openFile:

    data_sent_test = np.array(pickle.load(openFile), dtype=np.uint16)

with open(Q6_TEST_LABEL_ARR, 'rb') as openFile:

    data_labels_test = np.array(pickle.load(openFile), dtype=np.uint8).reshape((len(data_sent_test), 1))

with open(Q6_GLOVE_VOCAB, 'rb') as openFile:

    vocab = pickle.load(openFile)

with open(Q6_GLOVE_EMBEDDING, 'rb') as openFile:

    embedding = np.array(pickle.load(openFile), dtype=np.float32)

def chunks(claims, sentences, labels, n):

    for i in range(0, len(claims), n):

        yield claims[i:i + n], sentences[i:i+n], labels[i:i+n]
def cnn(lr=0.01, epochs=10, batch_size=10, test = False):

    dimensionality = 300

    max_length = data_claim.shape[1]

    if test:

        train_size = data_claim.shape[0] + data_claim_dev.shape[0]

    else:

        train_size = data_claim.shape[0]

#     filters = 300

    tf.reset_default_graph()



    values_1 = tf.placeholder(tf.int32)

    values_2 = tf.placeholder(tf.int32)

    values_3 = tf.placeholder(tf.float32)



    dataset = tf.data.Dataset.from_tensor_slices((values_1, values_2, values_3))

    dataset = dataset.repeat(epochs)

    batched_dataset = dataset.batch(batch_size)

    iterator = batched_dataset.make_initializable_iterator()

    claim_indices, sent_indices, y_true = iterator.get_next() # (batch_size, max_length), (batch_size, max_length), (batch_size,1)



    tf_embedding = tf.Variable(tf.constant(0.0, shape=[len(vocab), dimensionality]),

                    trainable=False)



    embedding_placeholder = tf.placeholder(tf.float32, [len(vocab), dimensionality])

    embedding_init = tf_embedding.assign(embedding_placeholder)



#     tf_word_ids = tf.placeholder(tf.int32, shape=[None, max_length])



    claim = tf.nn.embedding_lookup(

        params=tf_embedding,

        ids=claim_indices

    )



    sent = tf.nn.embedding_lookup(

        params=tf_embedding,

        ids=sent_indices

    )



    # shapes arrays for conv layers

    x1 = tf.reshape(claim, (-1, max_length, dimensionality)) 

    x2 = tf.reshape(sent, (-1, max_length, dimensionality))

    y_true = tf.reshape(y_true, (tf.shape(y_true)[0], 1))



    initializer = tf.contrib.layers.xavier_initializer()

    

    conv0 = tf.keras.layers.Conv1D(filters=100, kernel_size=(2), activation=tf.nn.relu, padding="same", \

                                   kernel_initializer=initializer, bias_initializer=initializer)

    conv1 = tf.keras.layers.Conv1D(filters=100, kernel_size=(3), activation=tf.nn.relu, padding="same", \

                                   kernel_initializer=initializer, bias_initializer=initializer)

    conv2 = tf.keras.layers.Conv1D(filters=100, kernel_size=(4), activation=tf.nn.relu, padding="same", \

                                   kernel_initializer=initializer, bias_initializer=initializer)

    conv3 = tf.keras.layers.Conv1D(filters=100, kernel_size=(5), activation=tf.nn.relu, padding="same", \

                                   kernel_initializer=initializer, bias_initializer=initializer)

    

    pool0 = tf.keras.layers.GlobalMaxPool1D()

    pool1 = tf.keras.layers.GlobalMaxPool1D()

    pool2 = tf.keras.layers.GlobalMaxPool1D()

    pool3 = tf.keras.layers.GlobalMaxPool1D()



    x1_rep = tf.concat([pool0(conv0.apply(x1)), pool1(conv1.apply(x1)), pool2(conv2.apply(x1)), pool3(conv3.apply(x1))], 1) # (batch_size, filters)

    x2_rep = tf.concat([pool0(conv0.apply(x2)), pool1(conv1.apply(x2)), pool2(conv2.apply(x2)), pool3(conv3.apply(x2))], 1) # (batch_size, filters)



    hidden = tf.keras.layers.Dense(10, activation='relu')

    y_h = hidden(tf.concat([x1_rep, x2_rep], 1)) # (batch_size, 10)



    linear = tf.keras.layers.Dense(1)

    logits = linear(y_h) # (50, 1)

    

    batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=logits)

    loss = tf.reduce_sum(batch_loss) # ()



    optimizer = tf.train.AdamOptimizer(lr)

    train = optimizer.minimize(loss)



    y_pred = tf.nn.sigmoid(logits) # (50, 1)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(y_pred), y_true), tf.float32))



    init = tf.global_variables_initializer() 

    saver = tf.train.Saver()



    sess = tf.Session()

    if test:

        sess.run(iterator.initializer, feed_dict={values_1: np.concatenate((data_claim, data_claim_dev), axis=0), 

                                                  values_2: np.concatenate((data_sent, data_sent_dev), axis=0), 

                                                  values_3: np.concatenate((data_labels, data_labels_dev), axis=0)})

    else:

        sess.run(iterator.initializer, feed_dict={values_1: data_claim, values_2: data_sent, values_3: data_labels})

    sess.run(init)

    sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

    

    dev_accuracy = []

    train_accuracy = []

    batches = int((train_size / batch_size))

    count_epochs = 0

    for i in range(batches * epochs):

        pred, true, _, loss_value, acc = sess.run((y_pred, y_true, train, loss, accuracy))

#         print("PREDICTIONS: ", pred)

#         print("TRUE VALUE", true)

#         print("LOSS", loss)

#         print("ACCURACY", acc)

        if (i+1) % batches == 0 and not test:

#             print("EPOCH:", count_epochs)

            count_epochs+=1

            # development accuracy

            total = 0

            count = 0

            for dev_claim, dev_sent, dev_labels in chunks(data_claim_dev, data_sent_dev, data_labels_dev, batch_size):

                dev_acc = sess.run(accuracy, feed_dict={claim_indices: dev_claim, sent_indices: dev_sent, y_true: dev_labels})

                total+=dev_acc

                count+=1

#             print("DEV: ", (total/count))

            dev_accuracy.append(total/count)

            # training accuracy

            total = 0

            count = 0

            for claims, sents, labels in chunks(data_claim, data_sent, data_labels, batch_size):

                train_acc = sess.run(accuracy, feed_dict={claim_indices: claims, sent_indices: sents, y_true: labels})

                total+=train_acc

                count+=1

#             print("TRAIN: ", (total/count))

            train_accuracy.append(total/count)

            

            if len(dev_accuracy) > 5 and dev_accuracy[-1] == dev_accuracy[-6]: # early stopping

                break

    if test: # testing results

        predictions = []

        for test_claim, test_sent, test_label in chunks(data_claim_test, data_sent_test, data_labels_test, batch_size):

            pred = sess.run(y_pred, feed_dict={claim_indices: test_claim, sent_indices: test_sent, y_true: test_label})

            predictions.extend(pred)

        predictions = np.array(predictions).flatten()

        true = np.array(data_labels_test).flatten()

        

        setting_string = f"lr-{lr}_e-{epochs}_ts-{train_size}"

        save_path = saver.save(sess, "./mymodel_" + setting_string + ".ckpt")

        #writing output to file

        filename = "output_" + str(lr) + "_" + str(epochs) + "_" + str(batch_size) + ".txt"

        with open(filename, "a") as output_file:

            output_file.write("Test predictions:\n")

            for i in predictions:

                output_file.write(str(i) + " ")

            output_file.write("\n")

            output_file.write("Test true:\n")

            for i in true:

                output_file.write(str(i) + " ")

            output_file.write("\n")

            

        # model results    

        rounded_predictions = np.around(predictions).astype(int)

        print("Accuracy:", accuracy_score(true, rounded_predictions))

        target_names = ['Non-Duplicate', 'Duplicate']

        print(classification_report(true, rounded_predictions, target_names=target_names))

        false_positive_rate, true_positive_rate, _ = roc_curve(true, predictions)

        area_under_curve = auc(false_positive_rate, true_positive_rate)

        plt.plot(false_positive_rate, true_positive_rate, label='Keras (area = {:.3f})'.format(area_under_curve))

        plt.xlabel('False positive rate')

        plt.ylabel('True positive rate')

        plt.title('ROC curve')

        plt.legend(loc='best')

        plt.show()

        return 0

    

    #writing output to file

    filename = "output_" + str(lr) + "_" + str(epochs) + "_" + str(batch_size) + ".txt"

    with open(filename, "w") as output_file:

        output_file.write("Development accuracy:\n")

        for i in dev_accuracy:

            output_file.write(str(i) + " ")

        output_file.write("\n")

        output_file.write("Train accuracy:\n")

        for i in train_accuracy:

            output_file.write(str(i) + " ")

        output_file.write("\n")



    setting_string = f"lr-{lr}_e-{epochs}_ts-{train_size}"

    save_path = saver.save(sess, "./mymodel_" + setting_string + ".ckpt")

    return dev_accuracy[-1]
def cv():

    lrs = [0.0001, 0.001]

    epochs = [20, 30]

    batch_sizes = [50, 100]

    max_acc = 0

    best_lr = 0

    best_epoch = 0

    best_batch_size = 0

    for lr in lrs:

        for epoch in epochs:

            for batch_size in batch_sizes:

                print("Trying", lr, epoch, batch_size)

                dev_acc = cnn(lr,epoch,batch_size)

                if dev_acc > max_acc:

                    print("Better found")

                    max_acc = dev_acc

                    best_lr = lr

                    best_epoch = epoch

                    best_batch_size = batch_size

    print("Best found is", best_lr, best_epoch, best_batch_size)

    cnn(best_lr, best_epoch, best_batch_size, test=True)
cv()