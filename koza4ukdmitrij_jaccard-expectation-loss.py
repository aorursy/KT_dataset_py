from PIL import Image

Image.open('../input/jelimages/Case1.jpg')
Image.open('../input/jelimages/Case2.jpg')
Image.open('../input/jelimages/Case3.jpg')
import numpy as np

import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.layers import Input, Dense, Concatenate

from tensorflow.keras.optimizers import SGD

from tensorflow.keras import models

tf.compat.v1.disable_eager_execution()



def jaccard_expectation_loss(y_true, y_pred):

    start_true, end_true = y_true[:, :MAX_LEN], y_true[:, MAX_LEN:]

    start_pred, end_pred = y_pred[:, :MAX_LEN], y_pred[:, MAX_LEN:]

    

    # for true labels we can use argmax() function, cause labels don't involve in SGD

    x_start = K.cast(K.argmax(start_true, axis=1), dtype=tf.float32)

    x_end   = K.cast(K.argmax(end_true  , axis=1), dtype=tf.float32)

    l = x_end - x_start + 1

    

    # some magic for getting indices matrix like this: [[0, 1, 2, 3], [0, 1, 2, 3]] 

    batch_size = K.shape(x_start)[0]

    ind_row = tf.range(0, MAX_LEN, dtype=tf.float32)

    ones_matrix = tf.ones([batch_size, MAX_LEN], dtype=tf.float32)

    ind_matrix = ind_row * ones_matrix

    

    # expectations for x_start^* (x_start_pred) and x_end^* (x_end_pred)

    x_start_pred = K.sum(start_pred * ind_matrix, axis=1)

    x_end_pred   = K.sum(end_pred   * ind_matrix, axis=1)

    

    relu11 = K.relu(x_start_pred - x_start)

    relu12 = K.relu(x_end   - x_end_pred  )

    relu21 = K.relu(x_start - x_start_pred)

    relu22 = K.relu(x_end_pred   - x_end  )

    

    intersection = l - relu11 - relu12

    union = l + relu21 + relu22

    jel = intersection / union

    

    return 1 - jel
MAX_LEN = 10

N_BATCH = 4



starts = np.array([np.eye(MAX_LEN)[1]] * N_BATCH).astype(float)

ends   = np.array([np.eye(MAX_LEN)[3]] * N_BATCH).astype(float)

y_true = np.concatenate([starts, ends], axis=1)



y_pred = np.random.rand(N_BATCH, 2 * MAX_LEN)
y_pred_inp = tf.compat.v1.placeholder(tf.float32, shape=[None, 2 * MAX_LEN])

y_true_inp = tf.compat.v1.placeholder(tf.float32, shape=[None, 2 * MAX_LEN])



jel = jaccard_expectation_loss(y_pred_inp, y_true_inp)



sess = tf.compat.v1.Session()

jel = sess.run(jel, feed_dict={y_pred_inp: y_pred, y_true_inp: y_true})

jel
def get_rand_discrete_output():

    output = np.zeros(MAX_LEN).astype(int)

    ind = np.random.randint(0, MAX_LEN - 1)

    output[ind] = 1

    return output



def get_discrete_batch(n_batch):

    starts_true, ends_true = [], []

    starts_pred, ends_pred = [], []



    batch_ind = 0

    while batch_ind < N_BATCH:

        starts_true_, ends_true_ = get_rand_discrete_output(), get_rand_discrete_output()

        starts_pred_, ends_pred_ = get_rand_discrete_output(), get_rand_discrete_output()

        if starts_true_.argmax() <= ends_true_.argmax():

            batch_ind += 1

            starts_true += [starts_true_]

            ends_true   += [ends_true_  ]

            starts_pred += [starts_pred_]

            ends_pred   += [ends_pred_  ]

    return np.array(starts_true), np.array(ends_true), np.array(starts_pred), np.array(ends_pred)
N_BATCH = 100



starts_true, ends_true, starts_pred, ends_pred = get_discrete_batch(N_BATCH)

y_true = np.concatenate([starts_true, ends_true], axis=1)

y_pred = np.concatenate([starts_pred, ends_pred], axis=1)
sess = tf.compat.v1.Session()

jel_arr = sess.run(

    jaccard_expectation_loss(y_pred_inp, y_true_inp),

    feed_dict={y_pred_inp: y_pred, y_true_inp: y_true}

)

len(jel_arr)
def jaccard(str1, str2):

    a = set(str(str1).lower().split()) 

    b = set(str(str2).lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))



def get_str_output(start, end):

    output = ((start.cumsum() - end[::-1].cumsum()[::-1]) == 0).astype(int)

    output = output * np.arange(1, len(start) + 1)

    return " ".join(output[output > 0].astype(str))
get_str_output(np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0]))
def checking_formula(eps=0.001):

    for batch_ind in range(N_BATCH):

        x_start0, x_end0 = starts_true[batch_ind].argmax(), ends_true[batch_ind].argmax()

        x_start1, x_end1 = starts_pred[batch_ind].argmax(), ends_pred[batch_ind].argmax()    



        str1 = get_str_output(starts_true[batch_ind], ends_true[batch_ind])

        if x_start1 <= x_end1:

            str2 = get_str_output(starts_pred[batch_ind], ends_pred[batch_ind])

        else:

            str2 = get_str_output(ends_pred[batch_ind], starts_pred[batch_ind])



        out1 = jaccard(str1, str2)

        out2 = 1 - jel_arr[batch_ind]



        if (x_start1 > x_end1):

            case = "incorrrect pred "

            assert out2 <= 0, (str1, str2, out1, out2)

        else:

            if (x_start1 > x_end0) or (x_start0 > x_end1):

                case = "intersection = 0"

                assert out2 <= 0, (str1, str2, out1, out2)

            else:

                case = "intersection > 0"

                assert abs(out1 - out2) < eps, (str1, str2, out1, out2)



        if case != "intersection > 0":

            show_str1 = str1.replace(" ", "").ljust(MAX_LEN)

            show_str2 = str2.replace(" ", "").ljust(MAX_LEN)

            print(f'[{case}]: true: {show_str1}, pred: {show_str2}, out1: {out1:.3f}, out2: {out2:.3f}') 
checking_formula(eps=0.001)
x1 = Input((MAX_LEN,), dtype=tf.int32)

x2 = Input((MAX_LEN,), dtype=tf.int32)



y1 = Dense(MAX_LEN)(x1)

y2 = Dense(MAX_LEN)(x2)



y = Concatenate(axis=1)([y1, y2])



model = models.Model(inputs=[x1, x2], outputs=y)

model.compile(loss=jaccard_expectation_loss, optimizer=SGD())
model.summary()
x1 = np.random.rand(N_BATCH, MAX_LEN)

x2 = np.random.rand(N_BATCH, MAX_LEN)



model.fit([x1, x2], y_true, epochs=10, verbose=1)