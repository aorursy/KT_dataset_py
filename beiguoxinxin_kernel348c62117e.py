import numpy as np

import tensorflow as tf
import os
label_1_data=np.load(r'../input/shuju_1.npy')

label_2_data=np.load(r'../input/shuju_2.npy')

label_3_data=np.load(r'../input/shuju_3.npy')

label_4_data=np.load(r'../input/shuju_4.npy')

label_5_data=np.load(r'../input/shuju_5.npy')

label_6_data=np.load(r'../input/shuju_6.npy')

label_7_data=np.load(r'../input/shuju_7.npy')
def zhuanhuadatat(data):

    import numpy as np

    data_1=data.reshape(1800,120,100,1)

    return data_1
data_1=zhuanhuadatat(label_1_data)

data_2=zhuanhuadatat(label_2_data)

data_3=zhuanhuadatat(label_3_data)

data_4=zhuanhuadatat(label_4_data)

data_5=zhuanhuadatat(label_5_data)

data_6=zhuanhuadatat(label_6_data)

data_7=zhuanhuadatat(label_7_data)
label=np.array([])

for j in range(7):

    ceshilabel=np.array([0,0,0,0,0,0,0])

    ceshilabel[j]=1

    for i in range(1800):

        label=np.append(label,ceshilabel)



label=label.reshape(-1,7)
label.shape
dataset=np.concatenate((data_1,data_2,data_3,data_4,data_5,data_6,data_7),axis=0)
def juzhihua(data):

    import numpy as np

    data[np.isnan(data)] = 0  # Nan数据进行消除

    mean_number=np.mean(data)

    std_data=np.std(data)

    data=(data-mean_number)/std_data

    return data

def guiyihua(data):

    import numpy as np

    data=(data-np.min(data))/(np.max(data)-np.min(data))

    return data
data=juzhihua(dataset)

data=guiyihua(dataset)
label=tf.cast(label,tf.int8)
tensor_train=tf.data.Dataset.from_tensor_slices(data)

tensor_label=tf.data.Dataset.from_tensor_slices(label)
ds_train=tf.data.Dataset.zip((tensor_train,tensor_label))
BATCH=64

ds_train=ds_train.shuffle(12600).repeat().batch(BATCH)
model=tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(120,100,1)),

                          tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

                           tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

                           tf.keras.layers.MaxPooling2D(),

                           tf.keras.layers.Conv2D(128,(3,3),activation='relu'),

                           tf.keras.layers.MaxPooling2D(),

                           tf.keras.layers.Conv2D(256,(3,3),activation='relu'),

                           tf.keras.layers.MaxPooling2D(),

                           tf.keras.layers.Conv2D(512,(3,3),activation='relu'),

                           tf.keras.layers.MaxPooling2D(),

                           tf.keras.layers.Conv2D(1024,(3,3),activation='relu'),

                           tf.keras.layers.GlobalAveragePooling2D(),

                           tf.keras.layers.Dense(256,activation='relu'),

                           tf.keras.layers.Dense(7,activation='softmax')

                          ])
model.summary()
model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['acc']

                )
history=model.fit(ds_train,epochs=100,steps_per_epoch=12600//64)  