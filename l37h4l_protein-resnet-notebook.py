import numpy as np
import pandas as pd
import tensorflow as tf
from skimage import io 
from skimage import transform
from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
dat_path = '../input'
print(os.listdir(dat_path))

# Any results you write to the current directory are saved as output.
train_img_path = dat_path + '/train'
test_img_path = dat_path + '/test'
label_path = dat_path + '/train.csv' 

#Data
labDat = pd.read_csv(label_path)

SplitRatio = 0.1 #Ratio to keep as validation and test data
datLen = len(labDat)

trainDat = labDat[ : int(datLen * (1 - 2 * SplitRatio))]
validDat = labDat[int(datLen * (1 - 2 * SplitRatio)) : int(datLen * (1 - SplitRatio))]
testDat = labDat[int(datLen * (1 - SplitRatio)) : ]

print('elements : ' + str(len(labDat)))
print('training : ' + str(len(trainDat)))
print('validation : ' + str(len(validDat)))
print('test : ' + str(len(testDat)))
#number of labels
num_labels = 28
class datGen():
    
    def make_batch(dat, path, batch_size, ind = 0, offset = 0, n_labels = 28, aug_chance = 0.3):
        I = np.identity(n_labels)
        imgs = []
        labels = []
        rnd_indexes = np.array(range(batch_size)) + ind * batch_size + offset
        for index in rnd_indexes:
            imgs.append(datGen.fetch_img('%s/%s' % (path, dat.loc[index][0]), aug_chance))
            labels.append(
                np.sum(I[np.array(dat.loc[index][1].split()).astype(np.int)], axis = 0) 
                ) # encodes labels into one-hot vectors and sums all the labels
        # returns normalized and flattened images along with the labels
        return np.reshape(np.array(imgs) / 256, [batch_size, 512 * 512 * 4]), np.array(labels)

    def fetch_img(path, aug_chance):
        img = []
        colors = ['red', 'green', 'blue', 'yellow']
        r = np.random.uniform()
        angle = 0
        if r < aug_chance:
            angle = np.random.uniform(90)
        for color in colors :
            img.append(datGen.augment(io.imread('%s_%s.png' % (path, color)), angle = angle))
        return np.stack(img, -1)
    
    def augment(img, angle = 0):
        if angle == 0:
            return img
        return transform.rotate(img, angle)
elements = 3
ex_dat = datGen.make_batch(labDat, train_img_path, elements, n_labels = num_labels, ind = 5, aug_chance = 0)
fig, ax = plt.subplots(elements, 4, sharex = 'col', sharey = 'row', figsize = [15,10], dpi = 150)
for ind, i in enumerate(ex_dat[0]):
    img = np.reshape(i, [512, 512, 4])
    ax[ind, 0].imshow(img[ : , : , 0], cmap = 'gnuplot')
    ax[ind, 1].imshow(img[ : , : , 1], cmap = 'hot')
    ax[ind, 2].imshow(img[ : , : , 2], cmap = 'magma')
    ax[ind, 3].imshow(img[ : , : , 3], cmap = 'terrain')
class ResNet():
    
    def resBlock(inp, out_space):
        rb1 = tf.layers.conv2d(inp, filters = out_space, kernel_size = 3, strides = 1, padding = 'same')
        rb2 = tf.layers.conv2d(rb1, filters = out_space, kernel_size = 3, strides = 1, padding = 'same')
        return tf.nn.relu(rb2 + inp)
    
    def __init__(self, input_size, out_space, res = 512, l_rate = 0.0001, beta = 0.09):
        # Placeholders
        self.inputVec = tf.placeholder(dtype = tf.float32, shape = [None, input_size])
        self.labels = tf.placeholder(dtype = tf.float32, shape = [None, out_space])
        
        # Convolutional layers
        self.X = tf.reshape(self.inputVec, shape = [-1, res, res, 4])
        self.cnv1 = tf.layers.conv2d(self.X, filters = 64, kernel_size = 7, strides = 2, padding = 'same')
        self.pool1 = tf.layers.max_pooling2d(self.cnv1, pool_size = 3, strides = 2, padding = 'same')
        
        self.res1 = ResNet.resBlock(self.pool1, 64)
        self.res2 = ResNet.resBlock(self.res1, 64)
        self.res3 = ResNet.resBlock(self.res2, 64)
        
        self.cnv2 = tf.layers.conv2d(self.res3, filters = 128, kernel_size = 3, strides = 2, padding = 'same')
        self.cnv2 = tf.layers.batch_normalization(self.cnv2)
        self.cnv3 = tf.layers.conv2d(self.cnv2, filters = 128, kernel_size = 3, strides = 1, padding = 'same')
        self.cnv3 = tf.layers.dropout(self.cnv3)
        
        self.res4 = ResNet.resBlock(self.cnv3, 128)
        
        self.cnv4 = tf.layers.conv2d(self.res4, filters = 256, kernel_size = 3, strides = 2, padding = 'same')
        self.cnv4 = tf.layers.batch_normalization(self.cnv4)
        self.cnv5 = tf.layers.conv2d(self.cnv4, filters = 256, kernel_size = 3, strides = 1, padding = 'same')
        self.cnv5 = tf.layers.dropout(self.cnv5)
        
        self.res5 = ResNet.resBlock(self.cnv5, 256)
        self.res6 = ResNet.resBlock(self.res5, 256)
        
        self.cnv6 = tf.layers.conv2d(self.res6, filters = 512, kernel_size = 3, strides = 2, padding = 'same')
        self.cnv6 = tf.layers.batch_normalization(self.cnv6)
        self.cnv7 = tf.layers.conv2d(self.cnv6, filters = 512, kernel_size = 3, strides = 2, padding = 'same')
        self.cnv7 = tf.layers.dropout(self.cnv7)
        
        self.res7 = ResNet.resBlock(self.cnv7, 512)
        self.res8 = ResNet.resBlock(self.res7, 512)
        
        self.pool2 = tf.layers.average_pooling2d(self.res8, pool_size = 3, strides = 2, padding = 'same')
        
        # Fully connected layer
        self.fc1 = tf.layers.flatten(self.pool2)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.fc1W = tf.Variable(xavier_init([8192, out_space]))
        
        self.logits = tf.matmul(self.fc1, self.fc1W)
        self.out = tf.sigmoid(self.logits)
        
        # Loss
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.reduce_mean(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.labels, logits = self.logits)) + beta * sum(reg_losses))
        self.optimizer = tf.train.AdamOptimizer(learning_rate = l_rate)
        self.updateOp = self.optimizer.minimize(self.loss)
#Hyperparameters 
batch_size = 40
learning_rate  = 0.0001
epochs = 10

#Create ops
Classifier = ResNet(512 * 512 * 4, 28, l_rate = learning_rate)
init = tf.global_variables_initializer()
saver = tf.train.Saver(tf.trainable_variables())

#Session
sess = tf.InteractiveSession()
tf.reset_default_graph()
sess.run(init)

if not os.path.exists('../checkpoints'):
    os.mkdir('../checkpoints')
try:
    saver.restore(sess, '../checkpoints/test_model.ckpt')
    print('...model loaded from previous checkpoint')
except:
    pass
tr_losses = []
vl_losses = []
for epoch in range(epochs):
    temp_t_loss = []
    temp_v_loss = []
    for i in range(int(datLen * (1 - 2 * SplitRatio)) // batch_size):
        trDat = datGen.make_batch(trainDat, train_img_path, batch_size = batch_size, ind = i)
        tr_loss, _ = sess.run([Classifier.loss, Classifier.updateOp], feed_dict = {Classifier.inputVec : trDat[0], Classifier.labels : trDat[1]})
        saver.save(sess, '../checkpoints/test_model.ckpt')
        
        if (i < int(datLen * SplitRatio) // batch_size):
            valDat = datGen.make_batch(validDat, train_img_path, batch_size = batch_size, ind = i, offset = int(datLen * (1 - 2 * SplitRatio)))
            vl_loss = sess.run([Classifier.loss], feed_dict = {Classifier.inputVec : valDat[0], Classifier.labels : valDat[1]})
            temp_v_loss.append(vl_loss)
        temp_t_loss.append(tr_loss)
        
        if (i % ((datLen // batch_size) // 20) == 0):            
            print('[epoch : %i , iter : %i] tr_loss : %f' % (epoch, i, tr_loss))
            
    #print training and validation loss 
    mean_t_loss = np.mean(temp_t_loss)
    mean_v_loss = np.mean(temp_v_loss)
    
    tr_losses.append(mean_t_loss)
    vl_losses.append(mean_v_loss)
    
    print('[epoch : %i] tr_loss : %f, vl_loss : %f' % (epoch, mean_t_loss, mean_v_loss))
    
sess.close()
plt.plot(tr_losses, label = 'training loss')
plt.plot(vl_losses, '--', label = 'validation loss')
plt.legend(loc = 'upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
def infer(inp_vector):
    out = sess.run(Classifier.out, feed_dict = {Classifier.inputVec : inp_vector})
    labels = []
    for element in out:
        label = np.squeeze(np.argwhere(element > 0.2)).astype(np.str)
        if label.size > 1:
            label = ' '.join(label)
        else :
            label = str(label)
        labels.append(label)
    return labels
    
inDat = datGen.make_batch(trainDat, train_img_path, batch_size = 4)
x = infer(inDat[0])
y = ['A', 'B', 'C', 'D']
list(zip(y,x))
#write to CSV
l = list(zip(y, x))
pd.DataFrame(l, columns = list('ky'))
import copy
testDF = copy.deepcopy(labDat[:200])
list(testDF.iloc[:,0])
class DFrameOps:
    
    def mkFrame():
        return pd.DataFrame({'Id' : [], 'Target' : []})
    
    def append(df, Id, Target):
        tempDF = pd.DataFrame(list(zip(Id, Target)), columns = ['Id', 'Target'])
        return df.append(tempDF)
    
    def resetIndex(df):
        return df.set_index(np.array(list(range(len(df)))))
        
tdf = DFrameOps.mkFrame()
inf_batch_size = 50
for i in range(len(testDF) // inf_batch_size):
    infDat, _ = datGen.make_batch(testDF, train_img_path, inf_batch_size, aug_chance = 0)
    infTarget = infer(infDat)
    infIds = testDF.iloc[i * inf_batch_size:(i + 1) * inf_batch_size, 0]
    tdf = DFrameOps.append(tdf, infIds, infTarget)
    
tdf = DFrameOps.resetIndex(tdf)
tdf[:10]
    

testDF[:10]
