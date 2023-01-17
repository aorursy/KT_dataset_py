# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import  datetime

date_depart=datetime.datetime.now()

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import weakref

import warnings



warnings.simplefilter("ignore")

import logging

logging.basicConfig(filename='python.log',level=logging.DEBUG)

logging.captureWarnings(True)

console = logging.StreamHandler()

console.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')

# tell the handler to use this format

console.setFormatter(formatter)

logging.getLogger('').addHandler(console)



from functools import partial

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import cv2

import matplotlib.pyplot as plt

import imageio

import imgaug

from imgaug import augmenters as iaa

from imgaug import parameters as iap

import keras

from keras.models import Sequential, Model

from keras.layers import Dense, Flatten, Activation, Dropout, GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers, applications

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

from keras import backend as K 

from concurrent import futures

import os

import json

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#!ls -Rlh ../input
duree_max=datetime.timedelta(hours=7,minutes=30)



fichier_modele_base="inaturalist" 

train_batch_size=64

full_train_batch_size=4

val_batch_size=4

epochs=int(1e8)



load_keras_weights=False

dmax=1024

dmin=600

dcrop=512

date_limite= date_depart+duree_max





ann_file = '../input/inaturalist-2019-fgvc6/train2019.json'

with open(ann_file) as data_file:

        train_anns = json.load(data_file)

def get_file_list(ann_file):

    with open(ann_file) as data_file:

            train_anns = json.load(data_file)



    train_anns_df = pd.DataFrame(train_anns['annotations'])[['image_id','category_id']]

    train_img_df = pd.DataFrame(train_anns['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})

    train_anns_df [train_anns_df.image_id.duplicated()]



    df_train=pd.merge(train_img_df,train_anns_df)[['category_id', 'file_name']]

    return  df_train





df_train=get_file_list(ann_file)

df_train
df_val=get_file_list('../input/inaturalist-2019-fgvc6/val2019.json')

df_val
train_anns.keys()
len(df_train.category_id.unique())
df_valid=get_file_list('../input/inaturalist-2019-fgvc6/train2019.json')

df_valid
classes=len(df_valid.category_id.unique())

print("random accuracy:",1/classes)

classes
df_train["catstr"]=df_train.category_id.astype("str")

p=next(df_train.sample(n=5).itertuples())

p.category_id

p.file_name
plt.figure(figsize=(25,25))

for n,tu in enumerate(df_train.sample(n=5).itertuples()):

    cat=tu.category_id

    im=tu.file_name

    plt.subplot(1,5,n+1)

    im=os.path.join("../input/inaturalist-2019-fgvc6/train_val2019/",im)

    plt.axis("off")

    plt.title(cat)

    plt.imshow(imageio.imread(im)) 
augmenters=[iaa.Sometimes(0.5,             

                                [iaa.Affine(scale=(0.99,1.05),

                                 translate_percent=(0,0.05), 

                                 rotate=iap.Normal(0,3),

                                 shear=iap.Normal(0,3),

                                 order=3)]),

                      iaa.Sometimes(0.3,[iaa.PiecewiseAffine(scale=(0,0.02))]),

                 

                                                        

   

                    iaa.Sometimes(0.1,

                                    [iaa.GaussianBlur(sigma=(0, 0.5)) ]),

                    iaa.Sometimes(0.1,

                                        [iaa.AverageBlur(k=(1, 2))]),

                    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),

                    iaa.AddElementwise((-5, 5)),

                     

                    iaa.Sometimes(0.1,

                                    [iaa.Superpixels(p_replace=(0.05, 0.8), n_segments=(16, 128))]

                                 ),

                    

            

            

            

                    iaa.Sometimes(0.3,

                                    [iaa.ElasticTransformation(alpha=(0, 5.0), sigma=(0.1,0.6) )]

                                 ),

                    

                    iaa.SaltAndPepper(p=(0.005,0.1)),

                                      iaa.Sometimes(0.4,

                                        [iaa.CoarseDropout(p=(0, 0.3),size_percent=(0.02, 0.5))]),



                        

                      

]



augmenters=[iaa.Sometimes(0.8,[iaa.Sequential(augmenters)])]
all(os.path.isfile(os.path.join("../input/inaturalist-2019-fgvc6/train_val2019/",f)) for f in df_train.sample(n=6000).file_name)
tu=next(df_train.itertuples())

tu.file_name

tu.category_id



df=df_train.copy()

df["req"]=None



def  imgaug_batch_gen(df,batch=16,executor=None,dmax=1024,dmin=512):

    if executor is None:

        executor=futures.ThreadPoolExecutor(max_workers=2)

    prefetch=int(batch*2.1+1) 

    df_len=len(df)

    def load_resize(f,dmax=dmax,dmin=dmin):

        img=imageio.imread(f,pilmode="RGB")

        resmin_img=np.min( img.shape[:2] )

        resmin=np.clip(resmin_img,dmin,dmax)

        r=resmin_img/resmin

            

        if r>1.2 or r<1.0:

            img=cv2.resize(img,None,fx=1.05/r, fy=1.05/r, interpolation = cv2.INTER_CUBIC)

        return img





    while True:

        

        df=df[["category_id","file_name"]].sample(frac=1).reset_index(drop=True)

        df["req"]=None

        i=0

        while i <(df_len-batch):

            df[df.req.notnull()].loc[:i]=None

            for j in range(prefetch):

                try:

                    if df.loc[j+i,"req"] is None:

                        f=os.path.join("../input/inaturalist-2019-fgvc6/train_val2019/",df.loc[j+i,"file_name"])

                        df.loc[j+i,"req"]=executor.submit(load_resize ,f)   

                except KeyError:

                    logging.exception("imgaug_batch_gen")



            df_batch=df.loc[i:i+batch-1]

            imgs=[req.result() for req in  df_batch.req]

            resmin=np.min(np.array([im.shape[:2]  for im in imgs]))

            resmin=np.clip(resmin,dmin,dmax)

            for j in range(batch):

                x,y,_=imgs[j].shape

                img_resmin=min(x,y)

                r=img_resmin/resmin

            

                if r>1.2 or r<1.0:

                    imgs[j]=cv2.resize(imgs[j],None,fx=1.05/r, fy=1.05/r, interpolation = cv2.INTER_CUBIC)

           

            categories=df_batch.category_id.values.astype("int32")                                        

            

            yield imgaug.imgaug.Batch(images=imgs,data=categories) 

            df.loc[i:i+batch-1,"req"]=None

            i+=batch

   





def  batch_gen(df,batch=16,augmenters=[],executor=None,dmax=1024,dmin=512,dcrop=512):

    if executor is None:

        executor=futures.ThreadPoolExecutor(max_workers=2)

    

    dmin=max(dmin,dcrop)

    dmax=max(dmin,dmax)

    gen =imgaug_batch_gen(df,batch=batch,executor=executor,dmax=dmax,dmin=dmin)

    aug=iaa.Sequential(augmenters+[iaa.CropToFixedSize(dmin,dmin),iaa.PadToFixedSize(dmin,dmin)])

    #aug_pool=aug.pool(processes=-1,maxtasksperchild=8)

    #gen=aug_pool.imap_batches_unordered(gen, chunksize=1)

    

    def aug_closure_gen(gen=gen):

        b=next(gen)

        fut=executor.submit(aug.augment_batches,[b],background=False)

        for b in gen:

            imgs=list(fut.result())[0].images_aug

            fut=executor.submit(aug.augment_batches,[b],background=False)                

            #b=list(aug.augment_batches([b],background=False))[0]



            

            imgs=[im[None,...] for im in imgs]

            X=np.concatenate(imgs).astype("float32")/256

            Y=b.data[...,None]      

            yield X,Y

   

    aug_closure=aug_closure_gen(gen)



    return aug_closure

            

        

        

    

    

                                                         

                                               

aug=iaa.Sequential(augmenters)

#aug_pool=aug.pool(processes=None, maxtasksperchild=None)



gen=batch_gen(df_train,batch=16,augmenters=augmenters)

X,Y=next(gen)

del gen

b_len=X.shape[0]

cols=4

rows=b_len//cols

if b_len%cols!=0:

    rows=rows+1

    

plt.figure(figsize=(25,25))

X.shape[0]

for n in range(b_len):

    

    cat=Y[n][0]



    plt.subplot(rows,cols,n+1)

    

    plt.axis("off")

    plt.title(cat)

    plt.imshow(X[n])      

    

    

pretrained_model=keras.applications.Xception(include_top=False, weights=None,pooling=None)

pretrained_model.load_weights("../input/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5")

fichier_modele=f"{fichier_modele_base}_{pretrained_model.name}.h5" 

fichier_modele
def mean_anwer_guess(y_true, y_pred):

    n=K.get_variable_shape(y_pred)[1]

    y_pred=K.cast(y_pred,"int64")

    preds=K.gather(y_true,y_pred)

    return K.mean(preds)





def mean_prandom_ratio(y_true, y_pred):

    n=K.get_variable_shape(y_pred)[1]

    y_pred=K.cast(y_pred,"int64")

    preds=K.gather(y_true,y_pred)

    return K.mean(preds*n)

import functools

def get_sparse_topn__categorical_accuracy(k):

    func=functools.partial(keras.metrics.sparse_top_k_categorical_accuracy,k=k)

    func.__name__=f"sparse_top_{k}_categorical_accuracy"

    return func
out_regulariser=keras.regularizers.l1_l2(l1=0.01, l2=0.05)



image_input=keras.Input(shape=(None,None,3), name="image_input", dtype="float32")

bottleneck1=pretrained_model(image_input)

bottleneck=keras.layers.Conv2D(filters=600,

                               kernel_size=3,padding="same", 

                               kernel_initializer=keras.initializers.Orthogonal(),

                               activation="selu",

                               activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.05),

                               strides=2

                              

                              )(bottleneck1)





pool1=(keras.layers.GlobalMaxPool2D()(bottleneck1))



pool=(keras.layers.GlobalMaxPool2D()(bottleneck))

pool=keras.layers.Concatenate()([pool,pool1])

pool=keras.layers.AlphaDropout(0.3)(pool)

pre_out=keras.layers.Dense(1200 ,

                           name="pre_out",

                            activation="selu",

                           kernel_regularizer=out_regulariser

                          )(pool)

out=keras.layers.Dense(classes,

                       activation="softmax"

                       ,name="out",

                       kernel_initializer=keras.initializers.Orthogonal(),

                       kernel_regularizer=out_regulariser)(pre_out)



model=keras.Model(inputs=image_input,outputs=out)

pretrained_model.trainable=False

optimizer=keras.optimizers.Adam(clipnorm=5. , clipvalue=5.,amsgrad=False,lr=0.0005)

model.compile(optimizer,

                            loss="sparse_categorical_crossentropy",

              metrics=["sparse_categorical_accuracy","sparse_categorical_crossentropy",mean_anwer_guess,get_sparse_topn__categorical_accuracy(2),

                       get_sparse_topn__categorical_accuracy(5),get_sparse_topn__categorical_accuracy(10)]

             )

model.summary()
class termination_date(keras.callbacks.Callback ):

    def __init__(self,end_date):

        self.end_date=end_date

    def on_epoch_end(self, batch, logs=None):

        if datetime.datetime.now()>self.end_date:

            self.model.stop_training = True

            logging.info("end date")

            

            

class logcallback(keras.callbacks.Callback):

    def __init__(self,logger=None):

        if logger is None:

            logger=logging.getLogger('traincallback')



        self.logger=logger

    def on_train_begin(self, logs={}):

        self.logger.info("training start: %s",self.model.name)

       



    def on_batch_end(self, batch, logs={}):

        met=""

        for k,v in logs.items():

            met+=f"{k}: {str(v)} "

        self.logger.debug("batch: %s - %s",batch,met)

        

    def on_epoch_end(self, epoch, logs=None):

        met=""

        for k,v in logs.items():

            met+=f"{k}: {str(v)} "

        self.logger.info("epoch: %s - %s",epoch,met)

    def on_train_end(self, logs={}):

        self.logger.info("training end: %s",self.model.name)

        

        

        

        



callbacks=[

        keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy',

                                          patience=10,

                                          min_delta=0.0005,

                                          factor=0.6,

                                          #min_lr=1e-6,

                                          verbose=1,

                                          cooldown=5



                                          ),

        keras.callbacks.ModelCheckpoint(monitor='val_sparse_categorical_accuracy',

                                        filepath=fichier_modele,

                                        verbose=1,

                                        save_best_only=True,

                                        period=20),

        keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',

                                      patience=20,

                                         

                                          verbose=1,

                                          restore_best_weights=True



                                          ),

         keras.callbacks.CSVLogger("train.csv", separator=',', append=True),

         termination_date(date_limite-datetime.timedelta(minutes=30)),

            keras.callbacks.BaseLogger(),

        logcallback()

    



        ]

if load_keras_weights:



    for fp in glob.glob(f"../input/**/{fichier_modele}",recursive=True):

        try:

            model.load_weights(fp, by_name=True, skip_mismatch=True)

            logging.info("loaded weights:",fb)

        except Exception as e:

            print(type(e),e)

            logging.exception("exception loading: %s %s",fp,e)

    if os.path.exists(fichier_modele):

        model.load_weights(fichier_modele, by_name=True, skip_mismatch=True)

        logging.info("loaded weights:",fb)


val_gen=batch_gen(df_val,batch=val_batch_size,dmax=dcrop,dmin=dcrop,dcrop=dcrop)
uptime=datetime.datetime.now()-date_depart

logging.info("pre train start %s",uptime)
hist_pre=model.fit_generator(batch_gen(df_train,batch=train_batch_size,augmenters=augmenters,dmax=dmax,dmin=dmin,dcrop=dcrop),

                     steps_per_epoch=1280/train_batch_size, 

                             epochs=150,

                             verbose=1,

                     validation_data=val_gen,

                     validation_steps=300/val_batch_size,

                     callbacks=   callbacks+    [ keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',

                                      patience=20,

                                         

                                          verbose=1,

                                          restore_best_weights=True



                                          )]

                     

                    

                   )

model.save(fichier_modele)



logging.info("pre train end %s",datetime.datetime.now()-date_depart)

logging.info("remaining time %s",datetime.timedelta(hours=9)+date_depart-datetime.datetime.now())
pretrained_model.trainable=True

optimizer=keras.optimizers.Adam(clipnorm=5. , clipvalue=5.,amsgrad=True)

model.compile(optimizer,

                            loss="sparse_categorical_crossentropy",

              metrics=model.metrics

             )

model.summary()
uptime=datetime.datetime.now()-date_depart

logging.info("main train %s",uptime)
hist=model.fit_generator(batch_gen(df_train,batch=full_train_batch_size,augmenters=augmenters,dmax=dmax,dmin=dmin,dcrop=dcrop),

                    steps_per_epoch=1280/full_train_batch_size, epochs=epochs, verbose=1,

                     validation_data=val_gen,

                     validation_steps=400/val_batch_size,

                         callbacks=   callbacks 

                        )

model.save(fichier_modele)

print(datetime.datetime.now()-date_depart)

                    
uptime=datetime.datetime.now()-date_depart

logging.info("post train %s",uptime)

logging.info("remaining time %s",datetime.timedelta(hours=9)+date_depart-datetime.datetime.now())
if (datetime.datetime.now()-date_depart)<datetime.timedelta(hours=8,minutes=30):

    fig=plt.figure(figsize=(15,7))

    plt.subplot("211")

    train_history=hist.history

    histories=[hist_pre.history,hist.history,hist_post]

    for k in train_history.keys():

        train_history[k]=[]

        for h in histories:

            train_history[k]+=h.get(k,[])





    for k in train_history.keys():

        if "acc" in k:

            plt.plot(train_history[k],label=k)

    plt.ylim(ymin=0.95,ymax=1.0)

    plt.legend()

    plt.subplot("212")

    plt.yscale("log")

    for k in train_history.keys():

        if "loss" in k:

            plt.plot(train_history[k],label=k)

    plt.legend()



    plt.ylim(ymax=0.8)

    fig.savefig("graph.png",dpi=200,transparent=False)

    #IPython.display.Image(filename="graph.png")
if (datetime.datetime.now()-date_depart)<datetime.timedelta(hours=7,minutes=30):

    for m,e in zip(model.metrics_names,

                   model.evaluate_generator(batch_gen(df_val,batch=64),steps=50, verbose=1) ):

                   print (m,e)
if (datetime.datetime.now()-date_depart)<datetime.timedelta(hours=7,minutes=30):



    ann_file = '../input/inaturalist-2019-fgvc6/test2019.json'

    with open(ann_file) as data_file:

            test_anns = json.load(data_file)







    df_test=pd.DataFrame(test_anns['images'])[["id","file_name"]]

    df_test
test_batch=50



if (datetime.datetime.now()-date_depart)<datetime.timedelta(hours=7,minutes=0): 

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_dataframe(      



            dataframe=df_test,    



            directory = "../input/inaturalist-2019-fgvc6/test2019",    

            x_col="file_name",

            target_size = (dcrop,dcrop),

            batch_size = test_batch,

            shuffle = False,

            class_mode = None

            )

    lengen=len(test_generator.filenames)

    #math.gcd(lengen)

    predict=model.predict_generator(test_generator, steps = len(test_generator.filenames)//test_batch+1,verbose=1)

    sub=np.argsort(predict)[:lengen,-10:]

    sub=np.flip(sub,1)

    df_test["predicted"]=[" ".join(str(n) for n in  pred  )for pred in sub  ]

    df_test["preds"]=df_test["predicted"]

   

    df_test[["id","preds", 'predicted']].to_csv("submission.csv", index=False)

    df_test

    

uptime=datetime.datetime.now()-date_depart

logging.info("end %s",uptime)

logging.info("remaining time %s",datetime.timedelta(hours=9)+date_depart-datetime.datetime.now())
