import tensorflow as tf

from keras.layers import Input, Dense

from keras.models import Model, Sequential

from keras import regularizers

from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve

from sklearn.preprocessing import MinMaxScaler

from sklearn.manifold import TSNE

import os



%matplotlib inline

np.random.seed(0)

tf.random.set_random_seed(123456)
### Utility Functions

## Plots

# Plot Feature Projection [credit: https://www.kaggle.com/shivamb/semi-supervised-classification-using-autoencoders]

def tsne_plot(x1, y1, name="graph.png"):

    tsne = TSNE(n_components=2, random_state=0)

    X_t = tsne.fit_transform(x1)



    plt.figure(figsize=(12, 8))

    plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', color='g', linewidth='1', alpha=0.8, label='Non Fraud')

    plt.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker='o', color='r', linewidth='1', alpha=0.8, label='Fraud')



    plt.legend(loc='best');

    plt.savefig(name);

    plt.show();

    

# Plot Keras training history

def plot_loss(hist):

    plt.plot(hist.history['loss'])

    plt.plot(hist.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()

    

## Util methods copied from OCAN package due to failure to install as custom package [credit:https://github.com/PanpanZheng/OCAN]

def xavier_init(size): # initialize the weight-matrix W.

    in_dim = size[0]

    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)

    return tf.random_normal(shape=size, stddev=xavier_stddev)



def sample_shuffle_uspv(X):

    n_samples = len(X)

    s = np.arange(n_samples)

    np.random.shuffle(s)

    return np.array(X[s])



def pull_away_loss(g):



    Nor = tf.norm(g, axis=1)

    Nor_mat = tf.tile(tf.expand_dims(Nor, axis=1),

                      [1, tf.shape(g)[1]])

    X = tf.divide(g, Nor_mat)

    X_X = tf.square(tf.matmul(X, tf.transpose(X)))

    mask = tf.subtract(tf.ones_like(X_X),

                       tf.diag(

                           tf.ones([tf.shape(X_X)[0]]))

                       )

    pt_loss = tf.divide(tf.reduce_sum(tf.multiply(X_X, mask)),

                        tf.multiply(

                            tf.cast(tf.shape(X_X)[0], tf.float32),

                            tf.cast(tf.shape(X_X)[0]-1, tf.float32)))



    return pt_loss



def one_hot(x, depth):

    x_one_hot = np.zeros((len(x), depth), dtype=np.int32)

    x = x.astype(int)

    for i in range(x_one_hot.shape[0]):

        x_one_hot[i, x[i]] = 1

    return x_one_hot



def sample_Z(m, n):   # generating the input for G.

    return np.random.uniform(-1., 1., size=[m, n])



def draw_trend(D_real_prob, D_fake_prob, D_val_prob, fm_loss, f1):



    fig = plt.figure()

    fig.patch.set_facecolor('w')

    # plt.subplot(311)

    p1, = plt.plot(D_real_prob, "-g")

    p2, = plt.plot(D_fake_prob, "--r")

    p3, = plt.plot(D_val_prob, ":c")

    plt.xlabel("# of epoch")

    plt.ylabel("probability")

    leg = plt.legend([p1, p2, p3], [r'$p(y|V_B)$', r'$p(y|\~{V})$', r'$p(y|V_M)$'], loc=1, bbox_to_anchor=(1, 1), borderaxespad=0.)

    leg.draw_frame(False)

    # plt.legend(frameon=False)



    fig = plt.figure()

    fig.patch.set_facecolor('w')

    # plt.subplot(312)

    p4, = plt.plot(fm_loss, "-b")

    plt.xlabel("# of epoch")

    plt.ylabel("feature matching loss")

    # plt.legend([p4], ["d_real_prob", "d_fake_prob", "d_val_prob"], loc=1, bbox_to_anchor=(1, 1), borderaxespad=0.)



    fig = plt.figure()

    fig.patch.set_facecolor('w')

    # plt.subplot(313)

    p5, = plt.plot(f1, "-y")

    plt.xlabel("# of epoch")

    plt.ylabel("F1")

    # plt.legend([p1, p2, p3, p4, p5], ["d_real_prob", "d_fake_prob", "d_val_prob", "fm_loss","f1"], loc=1, bbox_to_anchor=(1, 3.5), borderaxespad=0.)

    plt.show()



## OCAN TF Training Utils

def generator(z):

    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)

    G_logit = tf.nn.tanh(tf.matmul(G_h1, G_W2) + G_b2)

    return G_logit





def discriminator(x):

    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)

    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)

    D_logit = tf.matmul(D_h2, D_W3) + D_b3

    D_prob = tf.nn.softmax(D_logit)

    return D_prob, D_logit, D_h2





# pre-train net for density estimation.

def discriminator_tar(x):

    T_h1 = tf.nn.relu(tf.matmul(x, T_W1) + T_b1)

    T_h2 = tf.nn.relu(tf.matmul(T_h1, T_W2) + T_b2)

    T_logit = tf.matmul(T_h2, T_W3) + T_b3

    T_prob = tf.nn.softmax(T_logit)

    return T_prob, T_logit, T_h2
raw_data = pd.read_csv("../input/creditcard.csv")

data, data_test = train_test_split(raw_data, test_size=0.25)
raw_data_sample = data[data['Class'] == 0].sample(1000).append(data[data['Class'] == 1]).sample(frac=1).reset_index(drop=True)

raw_data_x = raw_data_sample.drop(['Class'], axis = 1)

raw_data_x[['Time']]=MinMaxScaler().fit_transform(raw_data_x[['Time']])

raw_data_x[['Amount']]=MinMaxScaler().fit_transform(raw_data_x[['Amount']])

tsne_plot(raw_data_x, raw_data_sample["Class"].values, "raw.png")
data.loc[:,"Time"] = data["Time"].apply(lambda x : x / 3600 % 24)

data.loc[:,'Amount'] = np.log(data['Amount']+1)



data_test.loc[:,"Time"] = data_test["Time"].apply(lambda x : x / 3600 % 24)

data_test.loc[:,'Amount'] = np.log(data_test['Amount']+1)

# data = data.drop(['Amount'], axis = 1)

print(data.shape)

data.head()
non_fraud = data[data['Class'] == 0].sample(1000)

fraud = data[data['Class'] == 1]



df = non_fraud.append(fraud).sample(frac=1).reset_index(drop=True)

X = df.drop(['Class'], axis = 1).values

Y = df["Class"].values



tsne_plot(X, Y, "original.png")
## input layer 

input_layer = Input(shape=(X.shape[1],))



## encoding part

encoded = Dense(100, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)

encoded = Dense(50, activation='sigmoid')(encoded) 



## decoding part

decoded = Dense(50, activation='tanh')(encoded)



## output layer

output_layer = Dense(X.shape[1], activation='relu')(decoded) 



# Autoencoder model

autoencoder = Model(input_layer, output_layer)



autoencoder.compile(optimizer="adadelta", loss="mse")



# Min-max scaling 

x = data.drop(["Class"], axis=1)

y = data["Class"].values



# x_scale = MinMaxScaler(feature_range=(-1, 1)).fit_transform(x)

x_norm, x_fraud = x.values[y == 0], x.values[y == 1]
checkpointer = ModelCheckpoint(filepath='bestmodel.hdf5', verbose=0, save_best_only=True)

earlystopper = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.005, patience=20, verbose=0, restore_best_weights=True)

x_norm_train_sample = x_norm[np.random.randint(x_norm.shape[0], size=10000),:]

hist = autoencoder.fit(x_norm_train_sample, x_norm_train_sample, 

                batch_size = 256, epochs = 400, 

                shuffle = True, validation_split = 0.05, verbose=0, callbacks=[checkpointer, earlystopper])

plot_loss(hist)
hidden_representation = Sequential()

hidden_representation.add(autoencoder.layers[0])

hidden_representation.add(autoencoder.layers[1])

hidden_representation.add(autoencoder.layers[2])



norm_hid_rep = hidden_representation.predict(x_norm[np.random.randint(x_norm.shape[0], size=700),:])

fraud_hid_rep = hidden_representation.predict(x_fraud)



# norm_hid_rep = MinMaxScaler(feature_range=(-1, 1)).fit_transform(norm_hid_rep)

# fraud_hid_rep = MinMaxScaler(feature_range=(-1, 1)).fit_transform(fraud_hid_rep)
rep_x = np.append(norm_hid_rep, fraud_hid_rep, axis = 0)

y_n = np.zeros(norm_hid_rep.shape[0])

y_f = np.ones(fraud_hid_rep.shape[0])

rep_y = np.append(y_n, y_f)

tsne_plot(rep_x, rep_y, "latent_representation.png")
from IPython.display import display, Image, HTML

display(HTML("""<table align="center">

<tr ><td><b>Actual Representation (Before) </b></td><td><b>Latent Representation (Actual)</b></td></tr>

<tr><td><img src='original.png'></td><td>

             <img src='latent_representation.png'></td></tr></table>"""))
dim_input = norm_hid_rep.shape[1]

mb_size = 70



D_dim = [dim_input, 100, 50, 2]

G_dim = [50, 100, dim_input]

Z_dim = G_dim[0]



X_oc = tf.placeholder(tf.float32, shape=[None, dim_input])

Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

X_tar = tf.placeholder(tf.float32, shape=[None, dim_input])



# define placeholders for labeled-data, unlabeled-data, noise-data and target-data.



X_oc = tf.placeholder(tf.float32, shape=[None, dim_input])

Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

X_tar = tf.placeholder(tf.float32, shape=[None, dim_input])

# X_val = tf.placeholder(tf.float32, shape=[None, dim_input])





# declare weights and biases of discriminator.



D_W1 = tf.Variable(xavier_init([D_dim[0], D_dim[1]]))

D_b1 = tf.Variable(tf.zeros(shape=[D_dim[1]]))



D_W2 = tf.Variable(xavier_init([D_dim[1], D_dim[2]]))

D_b2 = tf.Variable(tf.zeros(shape=[D_dim[2]]))



D_W3 = tf.Variable(xavier_init([D_dim[2], D_dim[3]]))

D_b3 = tf.Variable(tf.zeros(shape=[D_dim[3]]))



theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]







# declare weights and biases of generator.



G_W1 = tf.Variable(xavier_init([G_dim[0], G_dim[1]]))

G_b1 = tf.Variable(tf.zeros(shape=[G_dim[1]]))



G_W2 = tf.Variable(xavier_init([G_dim[1], G_dim[2]]))

G_b2 = tf.Variable(tf.zeros(shape=[G_dim[2]]))



theta_G = [G_W1, G_W2, G_b1, G_b2]





# declare weights and biases of pre-train net for density estimation.



T_W1 = tf.Variable(xavier_init([D_dim[0], D_dim[1]]))

T_b1 = tf.Variable(tf.zeros(shape=[D_dim[1]]))



T_W2 = tf.Variable(xavier_init([D_dim[1], D_dim[2]]))

T_b2 = tf.Variable(tf.zeros(shape=[D_dim[2]]))



T_W3 = tf.Variable(xavier_init([D_dim[2], D_dim[3]]))

T_b3 = tf.Variable(tf.zeros(shape=[D_dim[3]]))



theta_T = [T_W1, T_W2, T_W3, T_b1, T_b2, T_b3]



D_prob_real, D_logit_real, D_h2_real = discriminator(X_oc)



G_sample = generator(Z)

D_prob_gen, D_logit_gen, D_h2_gen = discriminator(G_sample)



D_prob_tar, D_logit_tar, D_h2_tar = discriminator_tar(X_tar)

D_prob_tar_gen, D_logit_tar_gen, D_h2_tar_gen = discriminator_tar(G_sample)

# D_prob_val, _, D_h1_val = discriminator(X_val)



# disc. loss

y_real= tf.placeholder(tf.int32, shape=[None, D_dim[3]])

y_gen = tf.placeholder(tf.int32, shape=[None, D_dim[3]])



D_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=D_logit_real,labels=y_real))

D_loss_gen = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=D_logit_gen, labels=y_gen))



ent_real_loss = -tf.reduce_mean(

                        tf.reduce_sum(

                            tf.multiply(D_prob_real, tf.log(D_prob_real)), 1

                        )

                    )



ent_gen_loss = -tf.reduce_mean(

                        tf.reduce_sum(

                            tf.multiply(D_prob_gen, tf.log(D_prob_gen)), 1

                        )

                    )



D_loss = D_loss_real + D_loss_gen + 1.85 * ent_real_loss





# gene. loss

pt_loss = pull_away_loss(D_h2_tar_gen)



y_tar= tf.placeholder(tf.int32, shape=[None, D_dim[3]])

T_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=D_logit_tar, labels=y_tar))

tar_thrld = tf.divide(tf.reduce_max(D_prob_tar_gen[:,-1]) +

                      tf.reduce_min(D_prob_tar_gen[:,-1]), 2)



indicator = tf.sign(

              tf.subtract(D_prob_tar_gen[:,-1],

                          tar_thrld))

condition = tf.greater(tf.zeros_like(indicator), indicator)

mask_tar = tf.where(condition, tf.zeros_like(indicator), indicator)

G_ent_loss = tf.reduce_mean(tf.multiply(tf.log(D_prob_tar_gen[:,-1]), mask_tar))



fm_loss = tf.reduce_mean(

            tf.sqrt(

                tf.reduce_sum(

                    tf.square(D_logit_real - D_logit_gen), 1

                    )

                )

            )



G_loss = pt_loss + G_ent_loss + fm_loss



D_solver = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(D_loss, var_list=theta_D)

G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

T_solver = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(T_loss, var_list=theta_T)





# Load data

# min_max_scaler = MinMaxScaler()



x_benign = norm_hid_rep # min_max_scaler.fit_transform(norm_hid_rep)

x_vandal = fraud_hid_rep # min_max_scaler.transform(fraud_hid_rep)



x_benign = sample_shuffle_uspv(x_benign)

x_vandal = sample_shuffle_uspv(x_vandal)



x_pre = x_benign



y_pre = np.zeros(len(x_pre))

y_pre = one_hot(y_pre, 2)



x_train = x_pre



y_real_mb = one_hot(np.zeros(mb_size), 2)

y_fake_mb = one_hot(np.ones(mb_size), 2)



x_test = x_benign.tolist() + x_vandal.tolist()

x_test = np.array(x_test)





y_test = np.zeros(len(x_test))



y_test[len(x_benign):] = 1





sess = tf.Session()

sess.run(tf.global_variables_initializer())



# pre-training for target distribution

_ = sess.run(T_solver,

             feed_dict={

                X_tar:x_pre,

                y_tar:y_pre

                })



q = np.divide(len(x_train), mb_size)



d_ben_pro, d_fake_pro, fm_loss_coll = list(), list(), list()

f1_score  = list()

d_val_pro = list()





n_round = 200



for n_epoch in range(n_round):



    X_mb_oc = sample_shuffle_uspv(x_train)



    for n_batch in range(int(q)):



        _, D_loss_curr, ent_real_curr = sess.run([D_solver, D_loss, ent_real_loss],

                                          feed_dict={

                                                     X_oc: X_mb_oc[n_batch*mb_size:(n_batch+1)*mb_size],

                                                     Z: sample_Z(mb_size, Z_dim),

                                                     y_real: y_real_mb,

                                                     y_gen: y_fake_mb

                                                     })



        _, G_loss_curr, fm_loss_curr = sess.run([G_solver, G_loss, fm_loss],

                                           feed_dict={Z: sample_Z(mb_size, Z_dim),

                                                      X_oc: X_mb_oc[n_batch*mb_size:(n_batch+1)*mb_size],

                                                      })



    D_prob_real_, D_prob_gen_ = sess.run([D_prob_real, D_prob_gen],

                                         feed_dict={X_oc: x_train,

                                                    Z: sample_Z(len(x_train), Z_dim)})





    D_prob_vandal_ = sess.run(D_prob_real,

                              feed_dict={X_oc:x_vandal})



    d_ben_pro.append(np.mean(D_prob_real_[:, 0]))

    d_fake_pro.append(np.mean(D_prob_gen_[:, 0]))

    d_val_pro.append(np.mean(D_prob_vandal_[:, 0]))

    fm_loss_coll.append(fm_loss_curr)



    prob, _ = sess.run([D_prob_real, D_logit_real], feed_dict={X_oc: x_test})

    y_pred = np.argmax(prob, axis=1)

    y_pred_prob = prob[:,1]

    conf_mat = classification_report(y_test, y_pred, target_names=['genuine', 'fraud'], digits=4)

    f1_score.append(float(list(filter(None, conf_mat.strip().split(" ")))[12]))

    # print conf_mat





draw_trend(d_ben_pro, d_fake_pro, d_val_pro, fm_loss_coll, f1_score)
print ("OCAN: ")

print(conf_mat)

print ("Accuracy Score: ", accuracy_score(y_test, y_pred))



train_x, val_x, train_y, val_y = train_test_split(x_test, y_test, test_size=0.4)



clf = LogisticRegression(solver="lbfgs").fit(train_x, train_y)

pred_y = clf.predict(val_x)

pred_y_prob = clf.predict_proba(val_x)[:,1]



print ("")

print ("Linear Classifier: ")

print (classification_report(val_y, pred_y, target_names=['genuine', 'fraud'], digits=4))

print ("Accuracy Score: ", accuracy_score(val_y, pred_y))



fpr, tpr, thresh = roc_curve(val_y, pred_y_prob)

auc = roc_auc_score(val_y, pred_y_prob)

fpr2, tpr2, thresh2 = roc_curve(y_test, y_pred_prob)

auc2 = roc_auc_score(y_test, y_pred_prob)

plt.plot(fpr,tpr,label="linear in-sample, auc="+str(auc))

plt.plot(fpr2,tpr2,label="OCAN in-sample, auc="+str(auc2))

plt.legend(loc='best')

plt.show()
test_hid_rep = hidden_representation.predict(data_test.drop(['Class'], axis = 1).values)

test_y = data_test["Class"].values



prob_test, _ = sess.run([D_prob_real, D_logit_real], feed_dict={X_oc: test_hid_rep})

y_pred_test = np.argmax(prob_test, axis=1)

y_pred_prob_test = prob_test[:,1]



conf_mat_test = classification_report(test_y, y_pred_test, target_names=['genuine', 'fraud'], digits=4)

print ("OCAN: ")

print(conf_mat_test)

print ("Accuracy Score: ", accuracy_score(test_y, y_pred_test))



pred_y_test = clf.predict(test_hid_rep)

pred_y_prob_test = clf.predict_proba(test_hid_rep)[:,1]



print ("")

print ("Linear Classifier: ")

print (classification_report(test_y, pred_y_test, target_names=['genuine', 'fraud'], digits=4))

print ("Accuracy Score: ", accuracy_score(test_y, pred_y_test))



fpr, tpr, thresh = roc_curve(test_y, pred_y_prob_test)

auc = roc_auc_score(test_y, pred_y_prob_test)

fpr2, tpr2, thresh2 = roc_curve(test_y, y_pred_prob_test)

auc2 = roc_auc_score(test_y, y_pred_prob_test)

plt.plot(fpr,tpr,label="linear out-of-sample, auc="+str(auc))

plt.plot(fpr2,tpr2,label="OCAN out-of-sample, auc="+str(auc2))

plt.legend(loc='best')

plt.show()