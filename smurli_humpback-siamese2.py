import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder



import os

print(os.listdir("../input"))
#define PATH and Constants

TRAIN_PATH="/kaggle/input/train"

TEST_PATH="/kaggle/input/test"





print("Files in Folder", TRAIN_PATH, len([name for name in os.listdir(TRAIN_PATH)]))

print("Files in Folder", TEST_PATH, len([name for name in os.listdir(TEST_PATH)]))



csv = pd.read_csv('../input/train.csv')

#Take new_whale out of training set

train_csv = csv[~(csv.Id == "new_whale")]

print(train_csv.head())

files = "../input/train/"+train_csv.Image.values

labels = train_csv.Id.values

print("files=",files[:5])

print("labels=",labels[:5])

le = LabelEncoder()

true_labels = le.fit_transform(train_csv["Id"]).astype(np.int32)

max_label_val = le.transform(le.classes_)[-1]



#print("Files with labels=\n",train_label_data[:5])

#le.inverse_transform([1])[0]
#Split data set to test and train.

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(files, true_labels, test_size=0.3)

print (x_train, y_train)

print (x_test, y_test)
import matplotlib.pyplot as plt

import skimage

from tqdm import tnrange, tqdm_notebook

from skimage.color import rgb2gray



def img_read_fn(file):

    image = skimage.io.imread(file,1)

    image = skimage.transform.resize(image, (128,128,1), anti_aliasing=True)

    #image = skimage.color.rgb2gray(image)

    return image



img = img_read_fn(x_train[6])

plt.imshow(img.reshape(128,128), cmap='gray')



print("Done!")
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D, Conv1D, MaxPooling1D

from keras.models import Model, Sequential

from keras.regularizers import l2

from keras import backend as K

from keras.optimizers import SGD,Adam

from keras.losses import binary_crossentropy

import numpy.random as rng

import numpy as np

import os

import pickle

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.utils import shuffle

%matplotlib inline



def W_init(shape,name=None):

    """Initialize weights as in paper"""

    values = rng.normal(loc=0,scale=1e-2,size=shape)

    return K.variable(values,name=name)

#//TODO: figure out how to initialize layer biases in keras.

def b_init(shape,name=None):

    """Initialize bias as in paper"""

    values=rng.normal(loc=0.5,scale=1e-2,size=shape)

    return K.variable(values,name=name)



input_shape = (128, 128, 1)

left_input = Input(input_shape)

right_input = Input(input_shape)

#build convnet to use in each siamese 'leg'

convnet = Sequential()

convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,

                   kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))

convnet.add(MaxPooling2D())

convnet.add(Conv2D(128,(7,7),activation='relu',

                   kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))

convnet.add(MaxPooling2D())

convnet.add(Conv2D(128,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))

convnet.add(MaxPooling2D())

convnet.add(Conv2D(256,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))

convnet.add(Flatten())

convnet.add(Dense(4096,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init))



#call the convnet Sequential model on each of the input tensors so params will be shared

encoded_l = convnet(left_input)

encoded_r = convnet(right_input)

#layer to merge two encoded inputs with the l1 distance between them

L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))

#call this layer on list of two input tensors.

L1_distance = L1_layer([encoded_l, encoded_r])

prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(L1_distance)

siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)



optimizer = Adam(0.00006)

#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking

siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)



siamese_net.count_params()



# Save model

model_json = siamese_net.to_json()

with open("/kaggle/working/model.json", "w") as json_file:

    json_file.write(model_json)

## Test pad

import random



h = 128

w = 128

batch_size = 32

N = 20

labels = random.sample(np.unique(y_train).tolist(), (N*2)-1)



test_image = np.empty([N,128,128,1])

support_set = np.empty([N,128,128,1])



# Make first entry to of same label in test_image and support_set

sameLabel = labels.pop(0)

t1 = np.where(y_train==sameLabel)[0]

if len(t1) == 1:

    #Only one image available for this label.

    #Duplicate the entry so it picks up same entry for left and right.

    np.append(t1, t1[0])

left = random.sample(t1.tolist(),1)[0]

right = random.sample(t1.tolist(),1)[0]       

test_image[0] = img_read_fn(x_train[left])

support_set[0] = img_read_fn(x_train[right])



#Make rest of the entries to be different for test_image and support_set

loop = 1

while loop < N:

    t1 = np.where(y_train==labels.pop(0))[0]

    index = random.sample(t1.tolist(),1)[0]

    test_image[loop] = img_read_fn(x_train[index])

    t1 = np.where(y_train==labels.pop(0))[0]

    index = random.sample(t1.tolist(),1)[0]

    support_set[loop] = img_read_fn(x_train[index])

    loop += 1

targets = np.zeros((N,))

targets[0] = 1



print(test_image.shape)

print(support_set.shape)

print(targets.shape)
from datetime import datetime



class Siamese_Loader:

    """For loading batches and testing tasks to a siamese net"""

    def __init__(self, x_train, y_train, x_val, y_val):

        self.x_train = x_train

        self.y_train = y_train

        self.x_val = x_val

        self.y_val = y_val

        

    def get_batch(self,batch_size,s="train"):

        """Create batch of n pairs, half same class, half different class"""

        if s == "train":

            X=self.x_train

            y=self.y_train

        else:

            X=self.x_val

            y=self.y_val

            

        random.seed(datetime.now())

        

        #n_classes, n_examples, w, h = X.shape

        h = 128

        w = 128

        pairs=[np.zeros((batch_size, h, w, 1)) for i in range(2)]

        targets=np.zeros((batch_size,))

        targets[:batch_size//2] = 1 #1 - Means same class

        

        #First half of image are of same class and next half is of different class

        count = 0

        for label in random.sample(np.unique(y).tolist(), int(batch_size/2)):

            t1 = np.where(y==label)[0]

            if len(t1) == 1:

                #Only one image available for this label. Duplicate the entry so it picks up same entry for left and right.

                np.append(t1, t1[0])

            left = random.sample(t1.tolist(),1)[0]

            right = random.sample(t1.tolist(),1)[0]       

            pairs[0][count,:,:,:] = img_read_fn(X[left])

            pairs[1][count,:,:,:] = img_read_fn(X[right])

            count += 1

    

        while (count < batch_size):

            label1  = random.sample(np.unique(y).tolist(), 1)

            label2 = random.sample(np.unique(y).tolist(), 1)

            #Makesure two labels are different

            while (label2 == label1):

                label2 = random.sample(np.unique(y).tolist(), 1)

            

            t1 = np.where(y==label1)[0]

            left = random.sample(t1.tolist(),1)[0]

            

            t2 = np.where(y==label2)[0]

            right = random.sample(t2.tolist(),1)[0]       

            pairs[0][count,:,:,:] = img_read_fn(X[left])

            pairs[1][count,:,:,:] = img_read_fn(X[right])

            count += 1

    

        return pairs, targets

    

    def generate(self, batch_size, s="train"):

        """a generator for batches, so model.fit_generator can be used. """

        while True:

            pairs, targets = self.get_batch(batch_size,s)

            yield (pairs, targets)    



    def make_oneshot_task(self,N,s="val"):

        """Create pairs of test image, support set for testing N way one-shot learning. """

        

        if s == "train":

            X=self.x_train

            y=self.y_train

        else:

            X=self.x_val

            y=self.y_val

            

        random.seed(datetime.now())

        

        labels = random.sample(np.unique(y).tolist(), (N*2)-1)



        test_image = np.empty([N,128,128,1])

        support_set = np.empty([N,128,128,1])



        # Make first entry to of same label in test_image and support_set

        sameLabel = labels.pop(0)

        t1 = np.where(y==sameLabel)[0]

        if len(t1) == 1:

            #Only one image available for this label.

            #Duplicate the entry so it picks up same entry for left and right.

            np.append(t1, t1[0])

        left = random.sample(t1.tolist(),1)[0]

        right = random.sample(t1.tolist(),1)[0]       

        test_image[0] = img_read_fn(X[left])

        support_set[0] = img_read_fn(X[right])



        #Make rest of the entries to be different for test_image and support_set

        loop = 1

        while loop < N:

            t1 = np.where(y==labels.pop(0))[0]

            index = random.sample(t1.tolist(),1)[0]

            test_image[loop] = img_read_fn(X[index])

            t1 = np.where(y==labels.pop(0))[0]

            index = random.sample(t1.tolist(),1)[0]

            support_set[loop] = img_read_fn(X[index])

            loop += 1

        targets = np.zeros((N,))

        targets[0] = 1

        

        targets, test_image, support_set = shuffle(targets, test_image, support_set)

        pairs = [test_image,support_set]

        return pairs, targets

    

    def test_oneshot(self,model,N,k,s="val",verbose=0):

        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""

        n_correct = 0

        if verbose:

            print("Evaluating model on {} random {} way one-shot learning tasks ...".format(k,N))

        for i in range(k):

            inputs, targets = self.make_oneshot_task(N,s)

            probs = model.predict(inputs)

            if np.argmax(probs) == np.argmax(targets):

                n_correct+=1

        percent_correct = (100.0*n_correct / k)

        if verbose:

            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))

        return percent_correct

    

    def train(self, model, epochs, verbosity):

        model.fit_generator(self.generate(batch_size),

                            

                             )

    

#Instantiate the class

loader = Siamese_Loader(x_train, y_train, x_test, y_test)
def concat_images(X):

    """Concatenates a bunch of images into a big matrix for plotting purposes."""

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

    ax1.matshow(pairs[0][0].reshape(128,128),cmap='gray')

    img = concat_images(pairs[1])

    ax1.get_yaxis().set_visible(False)

    ax1.get_xaxis().set_visible(False)

    ax2.matshow(img,cmap='gray')

    plt.xticks([])

    plt.yticks([])

    plt.show()

#example of a one-shot learning task

pairs, targets = loader.make_oneshot_task(20,"train")

plot_oneshot_task(pairs)
#Training loop

print("!")

evaluate_every = 10 # interval for evaluating on one-shot tasks

loss_every=1 # interval for printing loss (iterations)

batch_size = 32

n_iter = 1000#90000

N_way = 20 # how many classes for testing one-shot tasks>

n_val = 100 #250 #how mahy one-shot tasks to validate on?

best = -1

PATH='/kaggle/working'

weights_path = os.path.join(PATH, "weights.h5")

eval_value_path = os.path.join(PATH, "eval.value")

print("training")

for i in range(1, n_iter):

    (inputs,targets)=loader.get_batch(batch_size)

    loss=siamese_net.train_on_batch(inputs,targets)

    #print(loss)

    if i % loss_every == 0:

        print("iteration {}, training loss: {:.2f},".format(i,loss))



    if i % evaluate_every == 0:

        print("evaluating")

        val_acc = loader.test_oneshot(siamese_net,N_way,n_val,verbose=True)

        if val_acc >= best:

            print("saving")

            #siamese_net.save(weights_path)

            siamese_net.save_weights(weights_path)

            best=val_acc

            f = open(eval_value_path,'w')

            f.write(str(best))

            f.close()
