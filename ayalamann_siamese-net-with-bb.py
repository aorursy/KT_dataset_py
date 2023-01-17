import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import itertools

from random import shuffle

import matplotlib.pyplot as plt

import cv2

from PIL import Image as pil_image

from math import sqrt

import random

from keras.utils import Sequence

from keras.layers import Input

# from lapjv import lapjv

from tqdm import tqdm_notebook as tqdm

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.resnet50 import ResNet50,preprocess_input

from keras.applications.xception import Xception

from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers import *

from keras.models import Sequential, Model

from collections import OrderedDict

from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from keras.models import model_from_json, load_model

from keras import regularizers





print(os.listdir("../input"))
DATA="../input/humpback-whale-identification"

TRAIN_IMG="../input/humpback-whale-identification/train"

TEST_IMG="../input/humpback-whale-identification/test"

SUBMISSION_DF = '../input/pretrained-model/submission_siamese.csv'

BB_DATA="../input/bounding-boxes-using-image-processing"



SIAMESE_MOD="../input/siamese-trained-new/siamese_trained_bb.h5"

SIAMESE_MOD_MID="../input/siamese-trained-new/siamese_mid.hdf5"



SIMPLE_CNN_SUB="../input/cnn-outputs/submission_top100.csv"

TEST_PARTS_DIR="../input/pretrained-model"

IM_SIZE=100



TRAIN_FLG=True

# if os.path.isfile(SIAMESE_MOD):

#     TRAIN_FLG=False

#     print("model will be loaded from "+ SIAMESE_MOD)
test_df= pd.DataFrame({"Image":  os.listdir(TEST_IMG)})

print("test images:"+ str(len(test_df)))
train_df = pd.read_csv(os.path.join(DATA, 'train.csv'))

train_df=train_df[train_df['Image']!='859e1399e.jpg']

train_lbl=train_df.copy()

print("train images:"+ str(len(train_lbl)))

print("total unique class:"+ str(len(np.unique(train_lbl['Id']))))

train_lbl.head()
#take out whales with a single train example (2072 examples)

df=train_lbl.groupby(['Id']).size().reset_index(

    name='train_examples')

df=df[df['train_examples']>=2]

single_whale_set= set(df.Id.values)

print("number of classes with more than 1 examples:"+ str(len(df)))



# print("number of classes with more than 1 examples:"+ str(len(df)))

train_lbl=train_lbl[train_lbl['Id'].isin(df['Id'])]

print("number of train instances :"+ str(len(train_lbl)))
no_new_whale=train_lbl[train_lbl['Id']!='new_whale']

print("number of train instances :"+ str(len(no_new_whale)))
def fetch_whale_img_list(image_dir,labels):

    img_groups = {}

    for img_file in tqdm(labels["Image"].values,desc='fetch_whale_img_list'):

        pid=img_file

        train_itms=labels[labels['Image']==pid]

        if train_itms is not None and len(train_itms)>0:

            gid=labels[labels['Image']==pid].values[0][1] #this is the ralevant whale group

            if gid in img_groups:

                img_groups[gid].append(pid)

            else:

                img_groups[gid] = [pid]

    return img_groups



whales_train_list=fetch_whale_img_list(TRAIN_IMG,no_new_whale)
def get_random_image(img_groups, gname):

    photos = img_groups[gname]

    pname = np.random.choice(photos, size=1)[0]

    return pname

    

def create_triples(image_dir,labels,data_set='train',img_groups=whales_train_list):

#     img_groups = fetch_whale_img_list(image_dir,labels)

    # creat equal number of negative examples

    group_names = list(img_groups.keys())

    triples = []

    # positive pairs are any combination of images in same group

    for key in img_groups.keys():

        combs=itertools.combinations(img_groups[key], 2)

        triple_pos = [(x[0] , x[1] , 1) 

                 for x in combs]

        idx= np.random.choice(np.arange(len(triple_pos)),size=len(img_groups[key])-1,replace=False)

        triple_pos=[triple_pos[i] for i in idx]

        triples.extend(triple_pos)

        

        triple_neg = []

        for x in triple_pos:

            flg=True

            while flg:

                neg_w = np.random.choice(group_names, size=1, replace=False)[0]

                if neg_w!=key: 

                    flg=False

            right = get_random_image(img_groups, neg_w)

            triple_neg.append((x[0], right, 0))

        triples.extend(triple_neg)

#         print("added neg examples:"+str(len(triple_neg)))

#     for i in tqdm(range(len(pos_triples))):

#         g1, g2 = np.random.choice(np.arange(len(group_names)), size=2, replace=False)

#         left = get_random_image(img_groups, group_names, g1)

#         right = get_random_image(img_groups, group_names, g2)

#         neg_triples.append((left, right, 0))

#     pos_triples.extend(neg_triples)

    shuffle(triples)

    return triples
if TRAIN_FLG:

    triples_data = create_triples(TRAIN_IMG,no_new_whale)

    print(len(triples_data))

    print("triplets examples:")

    print(triples_data[0:5])
if TRAIN_FLG:

    #look at some examples

    imgs=triples_data[0:10]

    per_row=2

    rows=5

    cols = 2

    fig, axes = plt.subplots(rows,cols, figsize=(24//per_row*cols,24//per_row*rows))

    for ax in axes.flatten(): 

        ax.axis('off')



    left=0

    j=0

    for i,ax in enumerate(axes.flatten()):

        img=imgs[j][left]

        label=imgs[j][2]

        image_path=os.path.join(TRAIN_IMG, img)

        left=(left+1)%2

        if (i+1)%2==0:

            j=j+1

        ax.imshow(cv2.imread(image_path))

        ax.set_title(label)
#read BB data fo train and test

train_bb=pd.read_csv(os.path.join(BB_DATA,"boxs_train.csv"))

test_bb=pd.read_csv(os.path.join(BB_DATA,"boxs_test.csv"))

bb_all=train_bb.append(test_bb, ignore_index=True)



bb = {}

for i in range(len(bb_all)):

    image=bb_all['Image'].iloc[i].split("/")[-1]

    x0=float(bb_all['x0'].iloc[i])

    y0=float(bb_all['y0'].iloc[i])

    x1=float(bb_all['x1'].iloc[i])

    y1=float(bb_all['y1'].iloc[i])

    box=(x0,y0,x1,y1)

    #save to labels file and dir

#   croped_image= crop_img(image,box)

    bb[image] = box
# expnd boxes to compensate for bounding box errors

crop_margin = 0.1

def expand_bb(image,box,margin=crop_margin):

    size_x,size_y=image.shape[1],image.shape[0]

    x0, y0, x1, y1 = box[0],box[1],box[2],box[3]

    dx = x1 - x0

    dy = y1 - y0

    x0 = max(0,x0-dx * crop_margin)

    x1 = min(size_x,x1+ dx * crop_margin + 1)

    y0 = max(0,y0-dy * crop_margin)

    y1 = min(size_y, y1+dy * crop_margin + 1)

    return x0,y0,x1,y1



def crop_img(image,box):

    new_box=expand_bb(image,box)

    if len(image.shape)==3:

        new_image = image[int(new_box[1]):int(new_box[3]), int(new_box[0]):int(new_box[2]),:]

    else:

        new_image = image[int(new_box[1]):int(new_box[3]), int(new_box[0]):int(new_box[2])]

    return new_image
RESIZE_IMG = IM_SIZE

from skimage.transform import resize

from keras.utils import np_utils

import cv2



def read_crop_resize(image_path, boxes=bb):

    image = cv2.imread(image_path)

    if image is not None:

        if image_path.split("/")[-1] in boxes:

            box=boxes[image_path.split("/")[-1]]

            croped_image= crop_img(image,box)

            image=croped_image

        try:

            image = cv2.resize(image, (RESIZE_IMG, RESIZE_IMG)) 

        except cv2.error as e:

            print("error resizing image:"+image_path )

            return None

    return image



def preprocess_images(image_names, seed, datagen,directory=TRAIN_IMG):

    np.random.seed(seed)

#     X = np.zeros((len(image_names), RESIZE_IMG, RESIZE_IMG, 3))

    X = np.zeros((len(image_names), RESIZE_IMG, RESIZE_IMG,1))

    for i, image_name in enumerate(image_names):

        if os.path.isfile(image_name):

            image = read_crop_resize(image_name)

        else:

            image = read_crop_resize(os.path.join(directory, image_name))

        if image is not None:

            if datagen is not None:

                image = datagen.random_transform(image)

            else:

                image = image

            image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            image=np.expand_dims(image, axis=2)

            X[i]=image

#             X[i]=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        else:

            print("error reading image :"+image_name)

    return X



def image_triple_generator(image_triples, directory, batch_size,augment=True,shuffle=True):

    datagen_args = dict(rescale=1./255,rotation_range=10,

                        horizontal_flip=True)

    if not augment:

        datagen_args = dict(rescale=1./255)

    datagen_left = ImageDataGenerator(**datagen_args)

    datagen_right = ImageDataGenerator(**datagen_args)

#     image_cache = {}

    

    while True:

        # loop once per epoch

        num_recs = len(image_triples)

        if shuffle:

            indices = np.random.permutation(np.arange(num_recs))

        else:

            indices = np.arange(num_recs)

        num_batches = num_recs // batch_size

#         if num_recs % batch_size > 0: num_batches=+1

        for bid in range(num_batches):

            # loop once per batch

            batch_indices = indices[bid * batch_size : min(num_recs,(bid + 1) * batch_size)]

            batch = [image_triples[i] for i in batch_indices]

            # make sure image data generators generate same transformations

            seed = np.random.randint(low=0, high=1000, size=1)[0]

            Xleft = preprocess_images([b[0] for b in batch], seed, 

                                      datagen_left,directory)

            Xright = preprocess_images([b[1] for b in batch],seed,

                                       datagen_right, directory)

            Y = np.array([b[2] for b in batch]) # 0 or 1

            yield ([Xleft.astype(np.uint8), Xright.astype(np.uint8)], Y)

if TRAIN_FLG:

    triples_batch_gen = image_triple_generator(triples_data,TRAIN_IMG, 32)

    ([Xleft, Xright], Y) = triples_batch_gen.__next__()

    print("generator output shapes:")

    print(Xleft.shape, Xright.shape, Y.shape)
# plt.imshow(Xright[2])
def create_base_network(input_shape):

    '''Base network to be shared (eq. to feature extraction).

    '''

    seq = Sequential()

    # CONV => RELU => POOL

    seq.add(Conv2D(20, kernel_size=5, padding="same", input_shape=input_shape,data_format="channels_last",activation='relu'))

    seq.add(BatchNormalization())

    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    seq.add(BatchNormalization())

    # CONV => RELU => POOL

    seq.add(Conv2D(50, kernel_size=5, padding="same",activation='relu'))

    seq.add(BatchNormalization())

    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    seq.add(BatchNormalization())

    seq.add(Flatten())

    seq.add(Dense(500,kernel_regularizer=regularizers.l2(0.01)))

    

    return seq
if TRAIN_FLG:

    image_size=IM_SIZE

#     input_shape = (image_size,image_size, 3)

    input_shape = (image_size,image_size,1)

    base_network = create_base_network(input_shape)



#     input_shape = (image_size,image_size, 3)

    input_shape = (image_size,image_size,1)

    vector_left =Input(shape=base_network.output_shape[1:])

    vector_right = Input(shape=base_network.output_shape[1:])

    img_l = Input(shape=input_shape)

    img_r = Input(shape=input_shape)

    x_l         = base_network(img_l)

    x_r         = base_network(img_r)
if TRAIN_FLG:

    #layer to merge two encoded inputs with the l1 distance between them

    mid        = 32

    L_prod = Lambda(lambda x : x[0]*x[1])([vector_left, vector_right])

    L_sum = Lambda(lambda x : x[0] + x[1])([vector_left, vector_right])

    L1_distance= Lambda(lambda x : K.abs(x[0] - x[1]))([vector_left, vector_right])

    L2_distance= Lambda(lambda x : K.square(x[0] - x[1]))([vector_left, vector_right])

    distance= Concatenate()([L_prod, L_sum, L1_distance, L2_distance])

    distance= Reshape((4, base_network.output_shape[1], 1), name='reshape1')(distance)

    x = Conv2D(mid, (4, 1), activation='relu', padding='valid')(distance)

    x = BatchNormalization()(x)

    x = Reshape((base_network.output_shape[1], mid, 1))(x)

    x = Conv2D(1, (1, mid), activation='linear', padding='valid')(x)

    x = BatchNormalization()(x)

    x = Flatten(name='flatten')(x)

    pred = Dense(1, use_bias=True, activation='sigmoid', name='weighted-average',kernel_initializer="random_normal")(x)

    head_model = Model([vector_left, vector_right], outputs=pred, name='head')



    x = head_model([x_l, x_r])

    siamese_model = Model(inputs=[img_l, img_r], outputs= x)

    siamese_model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
if TRAIN_FLG:

    from keras.utils.vis_utils import plot_model

    plot_model(base_network, to_file='branch_plot.png', show_shapes=True, show_layer_names=True,expand_nested=True)

    pil_image.open('branch_plot.png')
if TRAIN_FLG:

    plot_model(head_model, to_file='head_plot.png', show_shapes=True, show_layer_names=True,expand_nested=True)

    pil_image.open('head_plot.png')
if TRAIN_FLG:

    triples_train,triples_test =train_test_split(triples_data, test_size=0.2, random_state=42)



    callbacks=[

        ReduceLROnPlateau(monitor='val_loss',patience=10,min_lr=1e-9,verbose=1,mode='min'),

        ModelCheckpoint('siamese_mid.hdf5',monitor='val_loss',save_best_only=True,verbose=1)

    ]



    BATCH_SIZE=32

    NUM_EPOCHS=30



    train_gen = image_triple_generator(triples_train,TRAIN_IMG, BATCH_SIZE)

    val_gen = image_triple_generator(triples_test,TRAIN_IMG, BATCH_SIZE)



    num_train_steps = len(triples_train) // BATCH_SIZE

    num_val_steps = len(triples_test) // BATCH_SIZE

#     num_train_steps = 100

#     num_val_steps = 30



    siamese_model.save('siamese_trained_bb.h5')

    history = siamese_model.fit_generator(train_gen,

                                  steps_per_epoch=num_train_steps,

                                  epochs=NUM_EPOCHS,

                                  validation_data=val_gen,

                                  validation_steps=num_val_steps,

                                          callbacks=callbacks)

    siamese_model.save('siamese_trained_bb.h5')

else: #load from file

    siamese_model=load_model(SIAMESE_MOD)

    siamese_model.load_weights(SIAMESE_MOD_MID)
PREDICT=False
# create pairs on test image,train image to gain similarity score between them

def create_pairs(test_img,train_imgs):

    pairs = []

    for img in train_imgs:

#         pair = (os.path.join(TEST_IMG,test_img) , os.path.join(TRAIN_IMG,img[0]) , 0)

        pair = (os.path.join(TEST_IMG,test_img) , os.path.join(TRAIN_IMG,img) , 0)

        pairs.append(pair)

    return pairs
import sys, os



# Disable

def blockPrint():

    sys.stdout = open(os.devnull, 'w')



# Restore

def enablePrint():

    sys.stdout = sys.__stdout__
test_paths=pd.read_csv(os.path.join(TEST_PARTS_DIR,"test_all.txt"),header=None, names=['Image'])

test_paths=test_paths['Image'].values

train_paths=train_df["Image"].values
if PREDICT:

    # scores is an array with number of claases dim

    #given a scores vector of size (len(train)), group train labels to top 5 scored whales 

    new_whale='new_whale'

    def get_top5whales(scores,train_paths=train_paths,threshold=0.99):

        vhigh = 0

        pos = [0, 0, 0, 0, 0, 0]



        top_w = []

        top5_submission=[]

        s = set()

        a = scores

        for j in list(reversed(np.argsort(scores))): # j is an encoded label of some whale

            img = train_paths[j] # get image value of train example j



            if a[j] < threshold and new_whale not in s: # if score is lower than threshold and we didn't put new whale yet, than put new whale in the list

                pos[len(top5_submission)] += 1

                s.add(new_whale)

                top5_submission.append(new_whale)

            if len(top5_submission) == 5: break;

            whales=no_new_whale[no_new_whale['Image']==img]["Id"]

            if whales is not None and len(whales)>0:

                for w in whales.values:

                    if w not in s: # if we didn't yet added this whale

                        if a[j] >= threshold:

                            vhigh += 1

                        s.add(w)

                        top5_submission.append(w)

            if len(top5_submission) == 5: break;

            if new_whale not in s: pos[5] += 1

        assert len(top5_submission) == 5 

        assert len(s) == 5

        return top5_submission
if PREDICT:

    # for each test image gain similarity core for each one of the train images, and group them to top 5 whales for submission

    batch_size=64



    submissions={}

    # test on small set:

    # test_paths_t=test_paths[:10]

    for i in tqdm(range(len(test_paths)),desc='scores'):

    #     print(i)

        tst_img=test_paths[i]

        pairs= create_pairs(tst_img,train_paths)

        pairs1=pairs[:batch_size*(len(pairs)//batch_size)]

        test_generator = image_triple_generator(image_triples=pairs1,directory=None, batch_size=batch_size,augment=False,shuffle=False)

        scors1= siamese_model.predict_generator(test_generator,verbose = 1,steps=len(pairs)//batch_size , workers=1) 

        scores=scors1.flatten()



        pairs2=pairs[batch_size*(len(pairs)//batch_size):]

        test_generator = image_triple_generator(image_triples=pairs2,directory=None, batch_size=len(pairs2),augment=False,shuffle=False)

        scors2= siamese_model.predict_generator(test_generator,verbose = 1,steps=1 , workers=1) 



        scores=np.concatenate((scores,scors2.flatten()))

        submissions[tst_img]=' '.join(get_top5whales(scores,train_paths=train_paths,threshold=0.99))

if PREDICT:

    df_whales=pd.DataFrame.from_dict(submissions, orient='index', columns=['Id'])

    df_whales['Image'] = df_whales.index

    df_whales.to_csv("whales_pred_siamese.csv", index = False) 
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



if PREDICT:

    # function that takes in a dataframe and creates a text link to  

    # download it (will only work for files < 2MB or so)

    # def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    def create_download_link(df, title = "Download CSV file",input_file='whales_pred_siamese.csv', filename = "data.csv"):  

        csv = pd.read_csv(input_file)

        csv= csv.to_csv(index = False)

        b64 = base64.b64encode(csv.encode())

        payload = b64.decode()

        html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

        html = html.format(payload=payload,title=title,filename=filename)

        return HTML(html)



    # create a random sample dataframe



    # create a link to download the dataframe

    create_download_link()
#load submission from file (calc offline)

if not PREDICT:

    df_whales=pd.read_csv(SUBMISSION_DF)

    df_whales.to_csv("whales_pred_siamese.csv", index = False) 