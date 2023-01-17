from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D

from keras.models import Model, Sequential

from keras.regularizers import l2

from keras import backend as K

from keras.optimizers import SGD,Adam

from keras.losses import binary_crossentropy

import sklearn.metrics as sm

import numpy.random as rng

import numpy as np

import pandas as pd

import os

import pickle

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.utils import shuffle

import os

%matplotlib inline
from keras.datasets import mnist

def loop(p,q,low,high):

  m=[]

  cat = {}

  n=0

  for j in range(low,high):

    r = np.where(q==j)

    cat[str(j)] = [n,None]

    t = np.random.choice(r[0],20)

    c = []

    for i in t:

      c.append(np.resize(p[i],(35,35)))

      n+=1

    cat[str(j)][1]=n-1

    m.append(np.stack(c))

  return m,cat



def createdataset(low,high,t):

  (a,b),(f,g)=mnist.load_data()

  m=[]

  cat = {}

  if t==0:

    m,cat=loop(a,b,low,high)

  else:

    m,cat=loop(f,g,low,high)

  m = np.stack(m)

  return m,cat



train , cat_train = createdataset(0,6,0)

test, cat_test = createdataset(6,10,0)



print(train.shape)

print(test.shape)

print(cat_train.keys())

print(cat_test.keys())
def W_init(shape,name=None,dtype=None):

    """Initialize weights as in paper"""

    values = rng.normal(loc=0,scale=1e-2,size=shape)

    return K.variable(values,name=name)

#//TODO: figure out how to initialize layer biases in keras.

def b_init(shape,name=None,dtype=None):

    """Initialize bias as in paper"""

    values=rng.normal(loc=0.5,scale=1e-2,size=shape)

    return K.variable(values,name=name)

nclass,nexample,row,col = train.shape

input_shape = (row,col, 1)

left_input = Input(input_shape)

right_input = Input(input_shape)

#build convnet to use in each siamese 'leg'

convnet = Sequential()

convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))

convnet.add(MaxPooling2D())

# convnet.add(Conv2D(128,(7,7),activation='relu',kernel_regularizer=l2(2e-4),kernel_initializer=W_init, bias_initializer=b_init))

# convnet.add(MaxPooling2D())

convnet.add(Conv2D(128,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))

# convnet.add(MaxPooling2D())

# convnet.add(Conv2D(256,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))

convnet.add(Flatten())

convnet.add(Dense(4096,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=W_init, bias_initializer=b_init))



#call the convnet Sequential model on each of the input tensors so params will be shared

encoded_l = convnet(left_input)

encoded_r = convnet(right_input)

#layer to merge two encoded inputs with the l1 distance between them

L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))

#call this layer on list of two input tensors.

L1_distance = L1_layer([encoded_l, encoded_r])

prediction = Dense(1,activation='sigmoid')(L1_distance)

siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

optimizer = Adam(0.00006)

#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking

siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)



siamese_net.count_params()
# nclass,nexample,row,col = train.shape

# input_shape = (row,col, 1)

# left_input = Input(input_shape)

# right_input = Input(input_shape)

# #build convnet to use in each siamese 'leg'

# convnet = Sequential()

# convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,kernel_regularizer=l2(2e-4)))

# convnet.add(MaxPooling2D())

# # convnet.add(Conv2D(128,(7,7),activation='relu',kernel_regularizer=l2(2e-4),kernel_initializer=W_init, bias_initializer=b_init))

# # convnet.add(MaxPooling2D())

# convnet.add(Conv2D(128,(4,4),activation='relu',kernel_regularizer=l2(2e-4)))

# # convnet.add(MaxPooling2D())

# # convnet.add(Conv2D(256,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))

# convnet.add(Flatten())

# convnet.add(Dense(4096,activation="sigmoid",kernel_regularizer=l2(1e-3)))



# #call the convnet Sequential model on each of the input tensors so params will be shared

# encoded_l = convnet(left_input)

# encoded_r = convnet(right_input)

# #layer to merge two encoded inputs with the l1 distance between them

# L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))

# #call this layer on list of two input tensors.

# L1_distance = L1_layer([encoded_l, encoded_r])

# prediction = Dense(1,activation='sigmoid')(L1_distance)

# siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

# siamese_net.load_weights("/kaggle/input/mnistweights/weights")

# optimizer = Adam(0.00006)

# #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking

# siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)



# siamese_net.count_params()
class Siamese_Loader:

    """For loading batches and testing tasks to a siamese net"""

    def __init__(self):

        self.data = {"train":train, "val":test}

        self.categories = {"train":cat_train, "val":cat_test}





    def get_batch(self,batch_size,s="train"):

        """Create batch of n pairs, half same class, half different class"""

        X=self.data[s]

        n_classes, n_examples, w, h = X.shape



        #randomly sample several classes to use in the batch

        categories = rng.choice(n_classes,size=(batch_size,),replace=False)

        #initialize 2 empty arrays for the input image batch

        pairs=[np.zeros((batch_size, h, w,1)) for i in range(2)]

        #initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class

        targets=np.zeros((batch_size,))

        targets[batch_size//2:] = 1

        for i in range(batch_size):

            category = categories[i]

            idx_1 = rng.randint(0, n_examples)

            pairs[0][i,:,:,:] = X[category, idx_1].reshape(w, h, 1)

            idx_2 = rng.randint(0, n_examples)

            #pick images of same class for 1st half, different for 2nd

            if i >= batch_size // 2:

                category_2 = category  

            else: 

                #add a random number to the category modulo n classes to ensure 2nd image has

                # ..different category

                category_2 = (category + rng.randint(1,n_classes)) % n_classes

            pairs[1][i,:,:,:] = X[category_2,idx_2].reshape(w, h,1)

        return pairs, targets



    def make_oneshot_task(self,N,s="val",language=None):

        """Create pairs of test image, support set for testing N way one-shot learning. """

        X=self.data[s]

        n_classes, n_examples, w, h = X.shape

        indices = rng.randint(0,n_examples,size=(N,))

        # if language is not None:

        #     low, high = self.categories[s][language]

        #     if N > high - low:

        #         raise ValueError("This language ({}) has less than {} letters".format(language, N))

        #     categories = rng.choice(range(low,high),size=(N,),replace=False)

            

        # else:#if no language specified just pick a bunch of random letters

        #     categories = rng.choice(range(n_classes),size=(N,),replace=False)      

        categories = rng.choice(range(n_classes),size=(N,),replace=True)      

        true_category = categories[0]

        ex1, ex2 = rng.choice(n_examples,replace=False,size=(2,))

        test_image = np.asarray([X[true_category,ex1,:,:]]*N).reshape(N, w, h,1)

        support_set = X[categories,indices,:,:]

        support_set[0,:,:] = X[true_category,ex2]

        support_set = support_set.reshape(N, w, h,1)

        targets = np.zeros((N,))

        targets[np.where(categories==true_category)] = 1

        targets, test_image, support_set = shuffle(targets, test_image, support_set)

        pairs = [test_image,support_set]



        return pairs, targets

    

    def test_oneshot(self,model,N,k,i,s="val",verbose=0):

        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""

        n_correct = 0

        if verbose:

            print("iteration no.{}".format(i))

            print("Evaluating model on {} random {} way one-shot learning tasks ...".format(k,N))

        sum = 0.0

        sum1  = 0.0

        for i in range(k):

            inputs, targets = self.make_oneshot_task(N,s)

            probs = model.predict(inputs)

            probability = []

            for i in probs:

                probability.append(round(i[0]))

            probability = np.array(probability)

            a = sm.confusion_matrix(targets,probability)

            if len(a)>1:

                sum+= sm.accuracy_score(targets,probability)

                sum1+= sm.f1_score(targets,probability)

            else:

                sum+=1.0

                sum1+=1.0

        percent = (sum / k)*100.0

        F1_score = sum1/k

        if verbose:

            print("Got an average of {}% Accuracy in {} way one-shot learning accuracy".format(round(percent,2),N))

            print("Got an average of {} F1-Score in {} way one-shot learning accuracy".format(round(F1_score,2),N))

        return percent,F1_score



#Instantiate the class

loader = Siamese_Loader()


def concat_images(X):

    """Concatenates a bunch of images into a big matrix for plotting purposes."""

    a,b,c,d = X.shape

    X = np.resize(X,(a,28,28,d))

    nc,h,w,_ = X.shape

    X = X.reshape(nc,h,w)

    n = np.ceil(np.sqrt(nc)).astype("int8")

    img = np.zeros((n*w,n*h))

    x = 0

    y = 0

    for example in range(nc):

        img[x*w:(x+1)*w,y*h:(y+1)*h] = X[example]

        y += 1

        if y >= n:

            y = 0

            x += 1

    return img





def plot_oneshot_task(pairs):

    """Takes a one-shot task given to a siamese net and  """

    fig,(ax1,ax2) = plt.subplots(2)

    ax1.matshow(np.resize(pairs[0][0],(28,28)),cmap='gray')

    img = concat_images(pairs[1])

    ax1.get_yaxis().set_visible(False)

    ax1.get_xaxis().set_visible(False)

    ax2.matshow(img,cmap='gray')

    plt.xticks([])

    plt.yticks([])

    plt.show()

#example of a one-shot learning task

pairs, targets = loader.make_oneshot_task(10,"train","0")

plot_oneshot_task(pairs)


#Training loop

os.chdir(r'/kaggle/working/')

print("!")

evaluate_every = 1 # interval for evaluating on one-shot tasks

loss_every=50 # interval for printing loss (iterations)

batch_size = 4

n_iter = 10000

N_way = 10 # how many classes for testing one-shot tasks>

n_val = 250 #how many one-shot tasks to validate on?

best = -1

s = -1

print("training")

for i in range(1, n_iter):

    (inputs,targets)=loader.get_batch(batch_size)

    loss=siamese_net.train_on_batch(inputs,targets)

    print("Loss is  = {}".format(round(loss,2)))

    if i % evaluate_every == 0:

        print("evaluating")

        val_acc,score = loader.test_oneshot(siamese_net,N_way,n_val,i,verbose=True)

        if val_acc >= best or s>= score :

            print("saving")

            siamese_net.save(r'weights')# weights_path = os.path.join(PATH, "weights")

            best=val_acc



    if i % loss_every == 0:

        print("iteration {}, training loss: {:.2f},".format(i,loss))
ways = np.arange(1,30,2)

resume =  False

val_accs, train_accs, valscore, trainscore = [], [], [], []

trials = 400

i=0

for N in ways:

    train,trains = loader.test_oneshot(siamese_net, N,trials,i, "train", verbose=True)

    val,vals = loader.test_oneshot(siamese_net, N,trials,i, "val", verbose=True)

    val_accs.append(val)

    train_accs.append(train)

    valscore.append(vals)

    trainscore.append(trains)

    i+=1
from statistics import mean

print("The Average testing Accuracy is {}%".format(round(mean(val_accs),2)))

print("The Average testing F1-Score is {}".format(round(mean(valscore),2)))



plt.figure(1)

plt.plot(ways,train_accs,"b",label="Siamese(train set)")

plt.plot(ways,val_accs,"r",label="Siamese(val set)")



plt.xlabel("Number of possible classes in one-shot tasks")

plt.ylabel("% Accuracy")

plt.title("MNIST One-Shot Learning performace of a Siamese Network")

# box = plt.get_position()

# plt.set_position([box.x0, box.y0, box.width * 0.8, box.height])

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()

# inputs,targets = loader.make_oneshot_task(10,"val")



plt.figure(2)

plt.plot(ways,trainscore,"g",label="Siamese(train set)")

plt.plot(ways,valscore,"r",label="Siamese(val set)")

plt.xlabel("Number of possible classes in one-shot tasks")

plt.ylabel("F1-Score")

plt.title("MNIST One-Shot Learning F1 Score of a Siamese Network")

# box = plt.get_position()

# plt.set_position([box.x0, box.y0, box.width * 0.8, box.height])

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()