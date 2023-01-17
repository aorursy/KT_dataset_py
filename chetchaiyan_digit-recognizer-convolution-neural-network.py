import pandas as pd
import tensorflow as tf
import numpy as np

train_dataset = pd.read_csv('../input/train.csv')
test_dataset = pd.read_csv('../input/test.csv')
print("Train Dataset")
print(train_dataset[:5])
print("Test Dataset")
print(test_dataset[:5])
print("Train Size : {}".format(train_dataset.shape[0]))
print("Test Size : {}".format(test_dataset.shape[0]))
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer

validation_ratio = 0.1

pixel_scaler = MinMaxScaler()
one_hot = LabelBinarizer()
one_hot.fit(range(10))
train_y = one_hot.transform(train_dataset['label'])
train_x = train_dataset.drop('label', axis=1)
print(train_x.values.shape)
train_x = pixel_scaler.fit_transform(train_x)
train_x = train_x.reshape( (-1, 28, 28, 1) )
test_x = pixel_scaler.transform(test_dataset)
test_x = test_x.reshape( (-1, 28, 28, 1) )

split = int(train_dataset.shape[0] * (1 - validation_ratio))

valid_x = train_x[split:train_x.shape[0]]
valid_y = train_y[split:train_x.shape[0]]
train_x = train_x[0:split]
train_y = train_y[0:split]
print("Train Size : {}".format(train_x.shape[0]))
print("Validation Size : {}".format(valid_x.shape[0]))
class DigitRecognizerCNN(object):
    def __init__(self, width=28, height=28, channel=1, label_size=10):
        self.width = width
        self.height = height
        self.channel = channel
        self.label_size = label_size
    
    def build_input(self):
        model_x = tf.placeholder(tf.float32, [None, self.width, self.height, self.channel])
        model_y = tf.placeholder(tf.int32, [None, self.label_size])
        model_lr = tf.placeholder(tf.float32)
        return model_x, model_y, model_lr
    
    def build_cnn(self, model_x, model_y):
        model_x = tf.reshape(model_x, (-1, 28, 28, 1))
        
        conv_1 = tf.layers.conv2d(model_x, 3, 16, padding='same', activation=tf.nn.relu)
        pooling_1 = tf.layers.max_pooling2d(conv_1, 2, 1, padding='same')
        conv_2 = tf.layers.conv2d(pooling_1, 3, 32, padding='same', activation=tf.nn.relu)
        pooling_2 = tf.layers.max_pooling2d(conv_2, 2, 1, padding='same')
        flatten = tf.layers.flatten(pooling_2)
        logits = tf.layers.dense(flatten, self.label_size, activation=None)
        return logits
    
    def build_output(self, model_y, logits):
        output = tf.argmax(logits, axis=1)
        loss = tf.losses.softmax_cross_entropy(model_y, logits)
        accuracy = tf.reduce_mean( tf.cast( tf.equal( tf.argmax(model_y, axis=1), output ), tf.float32) )
        return output, loss, accuracy
    
    def model_opt(self, model_lr, loss):
        opt = tf.train.AdamOptimizer(model_lr).minimize(loss)
        return opt
    
    def make_nn(self):
        tf.reset_default_graph()
        self.model_x, self.model_y, self.model_lr = self.build_input()
        self.logits = self.build_cnn(self.model_x, self.model_y)
        self.output, self.loss, self.accuracy = self.build_output(self.model_y, self.logits)
        self.opt = self.model_opt(self.model_lr, self.loss)
        
    def train(self, epoch, learning_rate, train_x, train_y, batch_size, valid_x, valid_y, get_batch_func, print_every):
        t_loss_list = []
        v_loss_list = []
        accuracy_list = []
        saver = tf.train.Saver()
        with tf.Session() as sess :
            sess.run(tf.global_variables_initializer())
            counter = 0
            for e in range(epoch) :
                for x,y in get_batch_func(train_x, train_y, batch_size):
                    feed_dict = {self.model_x:x, self.model_y:y, self.model_lr:learning_rate}
                    loss, _ = sess.run([self.loss, self.opt], feed_dict=feed_dict)
                    if counter % print_every == 0 :
                        feed_dict = {self.model_x:valid_x, self.model_y:valid_y, self.model_lr:learning_rate}
                        v_loss, accuracy = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
                        print("Epoch: {}/{}, Step: {}, T Loss: {:.4f}, V Loss: {:.4f}, Accuracy: {:.4f}".format(e+1, 
                                    epoch, counter, loss, v_loss, accuracy))
                        t_loss_list.append( loss )
                        v_loss_list.append( v_loss )
                        accuracy_list.append( accuracy )
                        save_path = saver.save(sess, "tensor/model.ckpt")
                    counter += 1
        return t_loss_list, v_loss_list, accuracy_list
    
    def predict(self, predict_x):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, "tensor/model.ckpt")
            feed_dict = {self.model_x:predict_x}
            output = sess.run(self.output, feed_dict=feed_dict)
            return output
# For Testing --- Delete after use
dr_cnn = DigitRecognizerCNN()
dr_cnn.make_nn()
def get_batch(feature, label, batch_size):
    num_batch = feature.shape[0] // batch_size
    for i in range(num_batch):
        start = i * batch_size
        end = (i + 1) * batch_size
        x = feature[start:end]
        y = label[start:end]
        yield x, y
learning_rate = 0.0001
batch_size = 32
epoch = 20
print_every = 1000

cnn = DigitRecognizerCNN()
cnn.make_nn()
t_loss, v_loss, acc = cnn.train(epoch, learning_rate, train_x, train_y, batch_size, valid_x, valid_y, get_batch, print_every)
cnn = DigitRecognizerCNN()
cnn.make_nn()
size = 2048
num_batch = len(test_x) // size + 1
predict = []
for i in range(num_batch):
    p = cnn.predict(test_x[i * size: (i+1) * size])
    predict = np.concatenate([predict, p])
print(predict.shape)
import csv

with open('predict.csv', 'w') as csv_file :
    writer = csv.writer(csv_file)
    writer.writerow(['ImageId', 'Label'])
    for i in range(len(predict)) :
        writer.writerow([str(i+1), str(int(predict[i]))])
    csv_file.close()
