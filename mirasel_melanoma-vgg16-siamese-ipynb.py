from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.applications import inception_v3,vgg16
from keras.layers import Lambda, merge
from keras.layers import AveragePooling2D,Conv2D,MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.losses import binary_crossentropy
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
from sklearn.utils import shuffle
from sklearn import preprocessing
from statistics import mean
import matplotlib.pyplot as plt
import pandas as pd
import numpy.random as rng
import numpy as np
import pickle
import random
import time
import cv2
import os
# # Defining data path
# IMAGE_PATH = "../input/siim-isic-melanoma-classification/"

# train_df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
# test_df = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

# images = [
#     train_df[(train_df['sex'].isin(['female'])) & (train_df['target']==0)]['image_name'].values,
#     train_df[(train_df['sex'].isin(['male'])) & (train_df['target']==0)]['image_name'].values,
#     train_df[(train_df['sex'].isin(['female'])) & (train_df['target']==1)]['image_name'].values,
#     train_df[(train_df['sex'].isin(['male'])) & (train_df['target']==1)]['image_name'].values
# ]

# # Extract 75 random images from every sex
# dataset = [
#     rng.choice(images[0]+'.jpg',size=(75,),replace=False), #random 75 female image for benign (50 for train,25 for test)
#     rng.choice(images[1]+'.jpg',size=(75,),replace=False), #random 75 male image for benign (50 for train,25 for test)
#     rng.choice(images[2]+'.jpg',size=(75,),replace=False), #random 75 female image for malignant (50 for train,25 for test)
#     rng.choice(images[3]+'.jpg',size=(75,),replace=False) #random 75 male image for malignant (50 for train,25 for test)
# ]
# def imgreadconvert(path):
#     img = cv2.imread(path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img,(224,224))
#     return img
# # Location of the image dir
# img_dir = IMAGE_PATH+'/jpeg/train'
# x_train1 = []
# x_train2 = []
# x_test1 = []
# x_test2 = []
# for i in range(75):
#     if i <50:
#         img = imgreadconvert(os.path.join(img_dir,dataset[0][i])) #female benign train
#         x_train1.append(img)
#         img = imgreadconvert(os.path.join(img_dir,dataset[1][i])) #male benign train
#         x_train1.append(img)
#         img = imgreadconvert(os.path.join(img_dir,dataset[2][i])) #female malignant train
#         x_train2.append(img)
#         img = imgreadconvert(os.path.join(img_dir,dataset[3][i])) #male malignant train
#         x_train2.append(img)
#     else:
#         img = imgreadconvert(os.path.join(img_dir,dataset[0][i])) #female benign test
#         x_test1.append(img)
#         img = imgreadconvert(os.path.join(img_dir,dataset[1][i])) #male benign test
#         x_test1.append(img)
#         img = imgreadconvert(os.path.join(img_dir,dataset[2][i])) #female malignant test
#         x_test2.append(img)
#         img = imgreadconvert(os.path.join(img_dir,dataset[3][i])) #male malignant test
#         x_test2.append(img)

# x_train1 = np.array(x_train1)
# x_train2 = np.array(x_train2)
# x_test1 = np.array(x_test1)
# x_test2 = np.array(x_test2)

# x_train1,x_train2 = shuffle(x_train1,x_train2)
# x_test1,x_test2 = shuffle(x_test1,x_test2)

# train_groups = [x_train1,x_train2]
# test_groups = [x_test1,x_test2]

# import pickle
# with open("melanoma.pickle","wb") as f:
#     pickle.dump((train_groups,test_groups),f)
import pickle
with open("../input/melanoma/melanoma.pickle", "rb") as f:
    (train_groups,test_groups) = pickle.load(f,encoding='latin1')
%matplotlib inline
def W_init(shape,name=None,dtype=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)
#//TODO: figure out how to initialize layer biases in keras.
def b_init(shape,name=None,dtype=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)

def buildmodel(input_shape,pretrain=None):
  left_input = Input(input_shape)
  right_input = Input(input_shape)
  #build convnet to use in each siamese 'leg'
  convnet = Sequential()
  if pretrain is None:
    convnet.add(Conv2D(32,(10,10),activation='relu'))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(64,(7,7),activation='relu',
                      kernel_regularizer=l2(2e-4),kernel_initializer=W_init, bias_initializer=b_init))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(64,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),
                      bias_initializer=b_init))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),
                      bias_initializer=b_init))
  else:
    pretrained = pretrain(input_shape=input_shape,include_top=False,weights=None,pooling=max)
    pretrained.trainable=False
    convnet.add(pretrained)

  convnet.add(Flatten())
  convnet.add(Dense(2048,activation="sigmoid",kernel_regularizer=l2(1e-3)))


  #call the convnet Sequential model on each of the input tensors so params will be shared
  encoded_l = convnet(left_input)
  encoded_r = convnet(right_input)
  #layer to merge two encoded inputs with the l1 distance between them
  L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
  #call this layer on list of two input tensors.
  L1_distance = L1_layer([encoded_l, encoded_r])
  prediction = Dense(3,activation='sigmoid')(L1_distance)
  siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
  optimizer = Adam(0.00006)
  #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking

  siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)

  siamese_net.count_params()
  siamese_net.summary()
  return siamese_net
class Siamese_Loader:
  """For loading batches and testing tasks to a siamese net"""
  def __init__(self,input):
    print("Siamese Loaded")
    self.input_shape = input
    print(self.input_shape)
      
  def get_batch(self,batch_size):
    """Create batch of n pairs, half same class, half different class"""

    h,w,c = self.input_shape

    #initialize 2 empty arrays for the input image batch
    pairs=[np.zeros((batch_size, h, w,c)) for i in range(2)]
    #initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
    targets=np.zeros((batch_size,))
    X1 = train_groups[0]
    X2 = train_groups[1]
    n_examples1= X1.shape[0]
    n_examples2= X2.shape[0]
    for i in range(batch_size):
      if i<(batch_size // 3):
        idx_1 = rng.choice(n_examples1,size=(2,),replace=False)
        pairs[0][i,:,:,:] = X1[idx_1[0]].reshape(h,w,c)
        pairs[1][i,:,:,:] = X1[idx_1[1]].reshape(h,w,c)
        targets[i] = 0
      elif i>=(batch_size // 3) and i < (batch_size // 3)+(batch_size // 3):
        idx_2 = rng.choice(n_examples2,size=(2,),replace=False)
        pairs[0][i,:,:,:] = X2[idx_2[0]].reshape(h,w,c)
        pairs[1][i,:,:,:] = X2[idx_2[1]].reshape(h,w,c)
        targets[i] = 1
      else:
        idx_1 = rng.randint(0, n_examples1)
        idx_2 = rng.randint(0, n_examples2)
        pairs[0][i,:,:,:] = X1[idx_1].reshape(h,w,c)
        pairs[1][i,:,:,:] = X2[idx_2].reshape(h,w,c)
        targets[i] = 2
    pairs[0],pairs[1],targets = shuffle(pairs[0],pairs[1],targets)
    return pairs, targets

  def make_oneshot_task(self,N,i):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    h,w,c = self.input_shape
    X1 = test_groups[0]
    X2 = test_groups[1]
    n_examples1= X1.shape[0]
    n_examples2= X2.shape[0]
    if i%2==0:
      ex = rng.choice(n_examples1,replace=False,size=(int(N/2),))
      test_image = np.asarray([X1[ex[0],:,:]]*N).reshape(N,h,w,c)
      support_set = X1[ex].reshape(int(N/2),h,w,c)
      targets = np.zeros((N,))
      idx = rng.choice(n_examples2,replace=False,size=(int(N/2),))
      support_set = np.append(support_set,X2[idx].reshape(int(N/2),h,w,c),axis=0)
      targets[int(N/2):] = 2
      targets,test_image, support_set = shuffle(targets, test_image, support_set)
      pairs = [test_image,support_set]
    else:
      ex = rng.choice(n_examples2,replace=False,size=(int(N/2),))
      test_image = np.asarray([X2[ex[0]]]*N).reshape(N,h,w,c)
      support_set= X2[ex].reshape(int(N/2),h,w,c)
      targets = np.ones((N,))
      idx = rng.choice(n_examples1,replace=False,size=(int(N/2),))
      support_set = np.append(support_set,X1[idx].reshape(int(N/2),h,w,c),axis=0)
      targets[int(N/2):] = 2
      targets, test_image, support_set = shuffle(targets, test_image, support_set)
      pairs = [test_image,support_set]
    
    return pairs, targets
  
  def test_oneshot(self,model,N,k):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    sum = 0.0
    sum1=0.0
    for i in range(k):
      inputs, targets = self.make_oneshot_task(N,i)
      probs = model.predict(inputs)
      probs = probs.argmax(axis=-1)
      sum+= accuracy_score(targets,probs)
      sum1+= f1_score(targets,probs,average='micro')
    percent = (sum / k)*100.0
    F1_score = sum1/k
    print("ACCURACY-{}% and F1-SCORE-{} in {} random {} way one-shot learning".format(round(percent,2),round(F1_score,2),k,N))
    return percent,F1_score
def plot_oneshot_task(pairs):
  """Takes a one-shot task given to a siamese net and  """
  plt.figure(1)
  plt.imshow(pairs[0][0])
  plt.title("Test Image")
  plt.axis('off')
  plt.figure(2)
  for i in range(pairs[1].shape[0]):
    plt.subplot(pairs[1].shape[0]//5,5,i+1)
    plt.imshow(pairs[1][i])
    plt.axis('off')
  plt.suptitle(str(pairs[1].shape[0])+" way one shot Support Image Set")
#example of a one-shot learning task

def train_model(siamese_net,ptm_name):
  #Training loop
  print("!")
  # evaluate_every = 1 # interval for evaluating on one-shot tasks
  # loss_every=50 # interval for printing loss (iterations)
  batch_size = 24
  n_iter = 10000
  N_way = 10 # how many classes for testing one-shot tasks>
  n_val = 25 #how mahy one-shot tasks to validate on?
  best = -1
  s=-1
  # weights_path = os.path.join(PATH, "weights")
  LOSS = [];ACC=[];F1SCORE=[]
  t=0.0
  print("training")
  for i in range(1, n_iter+1):
    print("----------------------------------------------------------------------------------")
    print("Iteration no-",i,"   (",int(t/3600),"hr,",int((t%3600)/60),"min,",int((t%3600)%60),"sec left)")
    start = time.time()
    (inputs,targets)=loader.get_batch(batch_size)
    targets = to_categorical(targets,num_classes=3)
    loss=siamese_net.train_on_batch(inputs,targets)
    print("Loss= ",loss)
    val_acc,score = loader.test_oneshot(siamese_net,N_way,n_val)
    if i%100==0 or i==1:
      LOSS.append(loss)
      ACC.append(val_acc)
      F1SCORE.append(score)
    end = time.time()
    if best < val_acc or s< score:
      if best < val_acc:
        best = val_acc
      if s< score:
        s = score
      print("saving")
      siamese_net.save(r'MelanomaWeights_'+ptm_name)
    t = (10000-i)*(end-start)
  return LOSS,ACC,F1SCORE

def train_performane(LOSS,ACC,F1SCORE,title):
    performance = {"loss":LOSS,"val_accuracy":ACC,"val_f1-score":F1SCORE}
    plt.figure(figsize=(15,4))
    x = np.arange(100,10001,100)
    x = np.insert(x, 0, 1)
    for i,j in zip(performance,range(1,4)):
        plt.subplot(1,3,j)
        plt.plot(x,performance[i])
        plt.xlabel("Iterations")
        plt.ylabel(i)
        plt.title(i+title)
    plt.show()
def testing(siamese_net):
  ways = np.arange(2,51,2)
  resume =  False
  val_accs, train_accs, valscore, trainscore = [], [], [], []
  trials = 100
  i=0
  for N in ways:
      train,trains = loader.test_oneshot(siamese_net, N,trials)
      val,vals = loader.test_oneshot(siamese_net, N,trials)
      val_accs.append(val)
      train_accs.append(train)
      valscore.append(vals)
      trainscore.append(trains)
      i+=1
  return val_accs, train_accs, valscore, trainscore
def test_result(val_accs, train_accs, valscore, trainscore,title):
  print("The Average testing Accuracy is {}%".format(round(mean(val_accs),2)))
  print("The Average testing F1-Score is {}".format(round(mean(valscore),2)))

  plt.figure(figsize=(15,4))
  plt.subplot(1,2,1)
  plt.plot(np.arange(2,51,2),train_accs,"b",label="Siamese(train set)")
  plt.plot(np.arange(2,51,2),val_accs,"r",label="Siamese(val set)")

  plt.xlabel("Number of possible classes in one-shot tasks")
  plt.ylabel("% Accuracy")
  plt.title(title+"Melanoma One-Shot Learning performace of a Siamese Network")
  # box = plt.get_position()
  # plt.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  plt.legend(loc='upper right', bbox_to_anchor=(1, 0.5))
  # inputs,targets = loader.make_oneshot_task(10,"val")

  plt.subplot(1,2,2)
  plt.plot(np.arange(2,51,2),trainscore,"g",label="Siamese(train set)")
  plt.plot(np.arange(2,51,2),valscore,"r",label="Siamese(val set)")
  plt.xlabel("Number of possible classes in one-shot tasks")
  plt.ylabel("F1-Score")
  plt.title(title+"Melanoma One-Shot Learning F1 Score of a Siamese Network")
  # box = plt.get_position()
  # plt.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  plt.legend(loc='upper right', bbox_to_anchor=(1, 0.5))
  plt.show()
input = (224,224,3)
loader = Siamese_Loader(input)
pairs, targets = loader.make_oneshot_task(10,1)
plot_oneshot_task(pairs)
model1 = buildmodel(input,vgg16.VGG16)
LOSS1,ACC1,F1SCORE1 = train_model(model1,"VGG16")
train_performane(LOSS1,ACC1,F1SCORE1,' of VGG16 Siamease Net')
val_accs1, train_accs1, valscore1, trainscore1 = testing(model1)
test_result(val_accs1, train_accs1, valscore1, trainscore1,"VGG16 ")