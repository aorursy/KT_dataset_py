import numpy as np

import os

import pandas as pd

import tensorflow as tf

!pip install efficientnet

from efficientnet import tfkeras as efn

#from skimage.color import rgb2hsv

import gc

#import cv2

size = (224,224)

batch_size = 4

from PIL import Image

# os.environ['KAGGLE_USERNAME'] = "kunduruanil" # username from the json file

# os.environ['KAGGLE_KEY'] = "8aeec45db3771cf1b773905fcc521e6e" # key from the json file

# !kaggle datasets download -d cdeotte/jpeg-melanoma-256x256 # api copied from kaggle

# os.listdir(os.getcwd())
# !unzip "jpeg-melanoma-256x256.zip"

# base = os.getcwd()

# os.listdir(base)
base = "../input/jpeg-melanoma-256x256/"

os.listdir(base)
train = pd.read_csv(base + "/train.csv")

print(train.shape)

train['sex']=train['sex'].replace({"male":1,"female":0})

train['age_approx']=train['age_approx'].fillna(train['age_approx'].mean())

train['anatom_site_general_challenge']=train['anatom_site_general_challenge'].fillna(train['anatom_site_general_challenge'].mode())

s = train['anatom_site_general_challenge'].value_counts()/train.shape[0]

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].replace(s.to_dict())

#td = train[['sex',"age_approx","anatom_site_general_challenge"]]

del s

gc.collect()

train.head()
df_0 = train[train['target']==1].sample(84,random_state=42)

df_1 = train[train['target']==0].sample(10000,random_state=42)

train = train.drop(df_0.index)

train = train.drop(df_1.index)

val=pd.concat([df_0,df_1])

val=val.reset_index()

print(val.shape)

td_val = val[['sex',"age_approx","anatom_site_general_challenge"]]

val_filenames = base + "/train/" + val["image_name"] +".jpg"

val_labels = val['target']

td_val.shape
df_0=train[train['target']==0].sample(15000,random_state=42)

df_1=train[train['target']==1]

train=pd.concat([df_0,df_1])

train=train.reset_index()

print(train.shape)

train_filenames = base + "/train/" + train["image_name"] +".jpg"

labels = train['target']

td_train = train[['sex',"age_approx","anatom_site_general_challenge"]]

print(td_train.shape)

del train,val

gc.collect()
val = pd.concat([val_filenames,val_labels ], axis=1)

train = pd.concat([train_filenames,labels ], axis=1)
class Mygenarator(tf.keras.utils.Sequence):

    

    def __init__(self,df,td,x_col,y_col=None,batch_size=2,num_classes=None,size=(224,224,3),shuffle=True):

        self.df = df

        self.td = td

        self.x_col = x_col

        self.y_col = y_col

        self.size = size

        self.indices = df.index.tolist()

        self.batch_size = batch_size

        self.num_classes = num_classes

        self.shuffle = shuffle

        self.on_epoch_end()

        

    def on_epoch_end(self):

        self.index = np.arange(len(self.indices))

        if self.shuffle == True:

            np.random.shuffle(self.index)

            

    def __len__(self):

     # Denotes the number of batches per epoch

        return len(self.indices) // self.batch_size

    

    

    def __getitem__(self, index):

        # Generate one batch of data

        # Generate indices of the batch

        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs

        batch = [self.indices[k] for k in index]

        # Generate data

        X, y = self.__get_data(batch)

        return X, y

    

    def __get_data(self, batch):

        # X.shape : (batch_size, *dim)

        # We can have multiple Xs and can return them as a list

        X1 = np.empty((self.batch_size,*self.size))

        X2 = np.empty((self.batch_size,3))

        y = np.empty((self.batch_size), dtype=int)

        # Generate data

        for i, id in enumerate(batch):

         # Store sample

            X1[i,] = self.read_img(self.df.loc[id,self.x_col])

            X2[i,] = self.td.loc[id,:].values

            y[i] = self.df.loc[id,self.y_col]

            

        return {"imgIn":X1,"tabIn":X2}, y

    

    def hair_removal(self,image):

      grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

      # kernel for morphologyEx

      kernel = cv2.getStructuringElement(1,(17,17))

      # apply MORPH_BLACKHAT to grayScale image

      blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

      # apply thresholding to blackhat

      _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)



      # inpaint with original image and threshold image

      final_image = cv2.inpaint(image,threshold,1,cv2.INPAINT_TELEA)

      final_image = cv2.cvtColor(final_image,cv2.COLOR_BGR2RGB)

      return final_image



    def read_img(self,file):

      #image = cv2.imread(file)

      #image = cv2.resize(image,self.size[:-1])

      im = np.array(Image.open(file).resize(size))/255.0

      #return self.hair_removal(image)

      return im
val_data = Mygenarator(df=val,td=td_val,x_col="image_name",y_col="target")

train_data = Mygenarator(df=train,td=td_train,x_col="image_name",y_col="target")
# dataset = tf.data.Dataset.from_tensor_slices((train_filenames, labels))

# def _parse_function(filename, label):

#     img = tf.io.read_file(filename)

#     img = tf.image.decode_jpeg(img, channels=3)

#     img = tf.image.resize(img, [*size])

#     img = tf.image.per_image_standardization(img)

#     img = tf.image.convert_image_dtype(img, tf.float32)

#     return img, label

# AUTOTUNE=tf.data.experimental.AUTOTUNE

# dataset = dataset.map(_parse_function,num_parallel_calls=AUTOTUNE)

# dataset = dataset.shuffle(buffer_size=10000,reshuffle_each_iteration=True)

# dataset = dataset.batch(batch_size)

# dataset = dataset.cache()
# val_data = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))

# val_data = val_data.map(_parse_function,num_parallel_calls=AUTOTUNE)

# val_data = val_data.shuffle(buffer_size=10000,reshuffle_each_iteration=True)

# val_data = val_data.batch(batch_size)

# val_data = val_data.cache()
class LossAndaucPrintingCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):

      msg = "\n"

      for k,v in logs.items():

        msg = msg + " " + str(k) + " : " + str(round(v,3))

      print(msg)
def get_model():

    model_input = tf.keras.Input(shape=(*size, 3), name='imgIn')

    tab_input = tf.keras.Input(shape=(3,),name="tabIn")

    dummy = tf.keras.layers.Lambda(lambda x:x)(model_input)

    outputs = []    

    for i in range(8):

        constructor = getattr(efn, f'EfficientNetB{i}')

 

        x = constructor(include_top=False, weights='imagenet', 

                        input_shape=(*size, 3), 

                        pooling='avg')(dummy)

        y = tf.keras.layers.Dense(100)(tab_input)

        y = tf.keras.layers.BatchNormalization()(y)

        y = tf.keras.layers.Activation("relu")(y)

        y = tf.keras.layers.Dropout(0.4)(y)

        y = tf.keras.layers.Dense(100)(y)

        y = tf.keras.layers.BatchNormalization()(y)

        y = tf.keras.layers.Activation("relu")(y)

        y = tf.keras.layers.Dropout(0.4)(y)

        concatenated = tf.keras.layers.concatenate([x, y], axis=-1)

        con =  tf.keras.layers.Dense(100, activation='relu')(concatenated)

        con = tf.keras.layers.BatchNormalization()(con)

        con = tf.keras.layers.Activation("relu")(con)

        con = tf.keras.layers.Dropout(0.4)(con)

        output = tf.keras.layers.Dense(1,name=f'Effnet{i}')(con)

        output = tf.keras.layers.Activation("sigmoid")(output)

        outputs.append(output)

 

    model = tf.keras.Model([model_input,tab_input], outputs, name='aNetwork')

    model.compile(optimizer='adam',loss = tf.keras.losses.BinaryCrossentropy(

    label_smoothing = 0.05),metrics=[tf.keras.metrics.Accuracy(),tf.keras.metrics.AUC(name='auc')])

    #tf.keras.metrics.AUC(name='auc')

    return model

model = get_model()

model.summary()
gc.collect()
#model.load_weights(save_path)
model.fit(train_data,epochs=1,validation_data=val_data, 

          callbacks=[tf.keras.callbacks.ModelCheckpoint(os.path.join(os.getcwd(),"effall"),   

                                                       save_weights_only=True), 

              tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3),LossAndaucPrintingCallback()])
test = pd.read_csv(base+"/test.csv")

test_files =  base + "/test/" + test["image_name"] +".jpg"
test_files[:-2].shape
testset = tf.data.Dataset.from_tensor_slices((test_files))

AUTOTUNE=tf.data.experimental.AUTOTUNE

def _parse(filename):

    img = tf.io.read_file(filename)

    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize(img, [*size])

    img = tf.image.per_image_standardization(img)

    img = tf.image.convert_image_dtype(img, tf.float32)

    return img

testset = testset.map(_parse,num_parallel_calls=AUTOTUNE)

testset = testset.batch(10)
pred = np.array([])

pred1 = np.array([])

for d in testset:

  a = np.array([i.reshape(-1,) for i in model.predict(d)])>=0.5

  pred1 = np.concatenate((pred,[1.0 if i>=4 else 0.0 for i in a.sum(axis=0)]),axis=None)

  pred = np.concatenate((pred,np.array([i.reshape(-1,) for i in model.predict(d)]).mean(axis=0)),axis=None)

pred.shape,pred1.shape
test['target']=pred1

test[['image_name',"target"]].to_csv(drive_path+"/subeffv1.csv",index=False)
test['target']=pred

test[['image_name',"target"]].to_csv(drive_path+"/subeffv2.csv",index=False)