

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.preprocessing import image

from tqdm import tqdm

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
nrows=None# for debug.

train=pd.read_csv("../input/train.csv",nrows=nrows)
train.info()


id_counts=train.Id.value_counts()

print("there are {} unique whale ids".format(train.Id.unique().size))

print("average images number:{}".format(id_counts.mean()))
def load_imgs(names,base_dir,target_size=[100,100],limit=-1):

    if limit!=-1:

        names=names[:limit]

    print("loading {} images".format(limit))

    N=len(names)

    X=np.zeros([N]+target_size+[3,])

    for i,name in enumerate(tqdm(names)):

        img=image.load_img(os.path.join(base_dir,name),target_size=target_size)

        X[i]=img

    return X/255

X=load_imgs(train.Image,"../input/train")

Y=pd.get_dummies(train.Id)

id_names=Y.columns

Y=Y.values

print("Dataset prepared!")
assert X.max()<=1,"images should in range [0,1]"
def topn(m:np.array,n:int):

    # from bigger to smaller.

    return np.argsort(-m,axis=1)[:,:n]

def map5(y_true:np.ndarray,y_pred:np.ndarray):

    """

    @param y_true: shape=[N,features] one hot encoding.

    @param y_pred: shape=[N,features] softmax probability

    @returns the competition metrics.

    """

    target_index=np.argmax(y_true,axis=1)

    top5_indexs=topn(y_pred,5)

    N=y_pred.shape[0]

    isin=np.zeros(N)

    for i in range(N):

        if target_index[i] in top5_indexs[i]:

            isin[i]=1

    return isin.mean().astype(np.float32)
print("test map5")

y1=np.random.randn(50000,100)

y2=np.random.randn(50000,100)

print("shold close to 0.05(5/100)")

map5(y1,y2)  # shold be close to 5/100
randomY=np.zeros([Y.shape[0],Y.shape[1]])

randomY[:,:]=Y.sum(axis=0)

random_score=map5(Y,randomY)

print("random score is:",random_score,",a good model should above this value")
from keras.layers import Conv2D,Dense,Input,BatchNormalization

from keras import layers,models,losses,optimizers,metrics

from keras import backend

import keras

import tensorflow as tf
def tf_map5(y_true,y_pred):

    """

    wappper of map5,to be a tensor operation

    @param y_true: tensor.

    @param y_pred: tensor.

    """

    return tf.py_func(map5,[y_true,y_pred],tf.float32)

    

    

classes=Y.shape[1]

def build_model():



    input_layer=layers.Input((100,100,3))

    x=layers.Conv2D(32,(3,3),padding="same",activation="relu")(input_layer)

    x=layers.Conv2D(32,(3,3),padding="same")(x)

    x=layers.BatchNormalization()(x)

    x=layers.Activation("relu")(x)

    x=layers.MaxPooling2D(pool_size=(2,2),strides=[2,2])(x)



    x=layers.Conv2D(64,(3,3),padding="same",activation="relu")(x)

    x=layers.Conv2D(64,(3,3),padding="same")(x)

    x=layers.BatchNormalization()(x)

    x=layers.Activation("relu")(x)

    x=layers.MaxPooling2D(pool_size=(2,2),strides=[2,2])(x)



    x=layers.Conv2D(128,(3,3),padding="same",activation="relu")(x)

    x=layers.Conv2D(128,(3,3),padding="same")(x)

    x=layers.BatchNormalization()(x)

    x=layers.Activation("relu")(x)

    x=layers.MaxPooling2D(pool_size=(2,2),strides=[2,2])(x)

    x=layers.Flatten()(x)

    x=layers.Dense(units=classes,activation="softmax")(x)



    model=models.Model(input_layer,x)

    return model

keras.backend.clear_session()

model=build_model()

# the learning_rate 0.001 is choose by fitting on the small batch(200).......



model.compile(loss=losses.categorical_crossentropy,optimizer=optimizers.SGD(lr=0.001,momentum=0.9),metrics=[tf_map5])

model.fit(X,Y,validation_split=0.2,epochs=3,batch_size=16)
# Train on 20288 samples, validate on 5073 samples

# Epoch 1/10

# 20288/20288 [==============================] - 141s 7ms/step - loss: 6.3519 - tf_map5: 0.3781 - val_loss: 6.0332 - val_tf_map5: 0.3860

# Epoch 2/10

# 20288/20288 [==============================] - 137s 7ms/step - loss: 5.3938 - tf_map5: 0.4115 - val_loss: 5.8058 - val_tf_map5: 0.4037

# Epoch 3/10

# 20288/20288 [==============================] - 137s 7ms/step - loss: 4.9662 - tf_map5: 0.4365 - val_loss: 5.7757 - val_tf_map5: 0.4140

# Epoch 4/10

# 20288/20288 [==============================] - 137s 7ms/step - loss: 4.3962 - tf_map5: 0.4833 - val_loss: 6.0837 - val_tf_map5: 0.3871

# Epoch 5/10

# 20288/20288 [==============================] - 137s 7ms/step - loss: 3.5756 - tf_map5: 0.5872 - val_loss: 5.8405 - val_tf_map5: 0.4193

# Epoch 6/10

# 20288/20288 [==============================] - 137s 7ms/step - loss: 2.6407 - tf_map5: 0.7620 - val_loss: 6.5429 - val_tf_map5: 0.3964

# Epoch 7/10

# 20288/20288 [==============================] - 137s 7ms/step - loss: 1.6129 - tf_map5: 0.8979 - val_loss: 7.5491 - val_tf_map5: 0.3830

# Epoch 8/10

# 20288/20288 [==============================] - 137s 7ms/step - loss: 0.9963 - tf_map5: 0.9485 - val_loss: 9.0888 - val_tf_map5: 0.2567

# Epoch 9/10

#  3520/20288 [====>.........................] - ETA: 1:46 - loss: 0.6372 - tf_map5: 0.9707
import os

test_names=os.listdir("../input/test")

limit=-1

if limit!=-1:

    test_names=test_names[:limit]
import gc

del X

del Y

gc.collect()
testX=load_imgs(test_names,"../input/test")
testY=model.predict(testX)
print("make submission data frame")

testY_5=topn(testY,5)

ID5=[] # the five id names

for i in range(len(testY_5)):

    names=[]

    for j in range(5):

        names.append(id_names[testY_5[i][j]])

    ID5.append(" ".join(names))

df=pd.DataFrame()

df["Image"]=test_names

df["Id"]=ID5

    
del testX

del testY

gc.collect()
df.head()
id_counts[:10]
df.to_csv("submission.csv",index=False)
!head -5 submission.csv