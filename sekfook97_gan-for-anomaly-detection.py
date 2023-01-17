# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# importing libraries

%matplotlib inline

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf

import numpy as np





from tensorflow.keras import layers

import time





from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
dtypes = {

"duration": np.int8,

"protocol_type": np.object,

"service": np.object,

"flag": np.object,

"src_bytes":  np.int8,

"dst_bytes":  np.int8,

"land": np.int8,

"wrong_fragment":  np.int8,

"urgent": np.int8,

"hot": np.int8,

"m_failed_logins":  np.int8,

"logged_in":  np.int8,

"num_compromised":  np.int8,

"root_shell":  np.int8,

"su_attempted":  np.int8,

"num_root": np.int8,

"num_file_creations":  np.int8,

"num_shells":  np.int8,

"num_access_files":  np.int8,

"num_outbound_cmds":  np.int8,

"is_host_login":  np.int8,

"is_guest_login":  np.int8,

"count": np.int8,

"srv_count":  np.int8,

"serror_rate": np.float16,

"srv_serror_rate": np.float16,

"rerror_rate": np.float16,

"srv_rerror_rate": np.float16,

"same_srv_rate": np.float16,

"diff_srv_rate": np.float16,

"srv_diff_host_rate": np.float16,

"dst_host_count":  np.int8,

"dst_host_srv_count":  np.int8,

"dst_host_same_srv_rate": np.float16,

"dst_host_diff_srv_rate": np.float16,

"dst_host_same_src_port_rate": np.float16,

"dst_host_srv_diff_host_rate": np.float16,

"dst_host_serror_rate": np.float16,

"dst_host_srv_serror_rate": np.float16,

"dst_host_rerror_rate": np.float16,

"dst_host_srv_rerror_rate": np.float16,

"label": np.object

}



columns = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","m_failed_logins",

"logged_in", "num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files",

"num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",

"same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",

"dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",

"dst_host_srv_rerror_rate","label"]



df = pd.read_csv("/kaggle/input/kdd-cup-1999-data/kddcup.data_10_percent_corrected", sep=",", names=columns, dtype=dtypes, index_col=None)
df.label.value_counts()
#Feature Selection

new_features=['dst_bytes',

 'logged_in',

 'count',

 'srv_count',

 'dst_host_count',

 'dst_host_srv_count',

 'dst_host_same_srv_rate',

 'dst_host_same_src_port_rate','label']
# 0 for normal data and 1 for abnormalities

df.label=df.label.apply(lambda x: 0 if x == 'normal.' else 1)
df=df[new_features]
#label encoding 

for column in df.columns:

    if df[column].dtype == np.object:

        encoded = LabelEncoder()

        

        encoded.fit(df[column])

        df[column] = encoded.transform(df[column])
#randomly sample 500 data point for training

df_train=df[df.label==0].sample(500)
#remove the item sampled from our dataset

index_list=df_train.index

df=df.drop(index_list)
#drop the label columns

df_train=df_train.drop('label',axis=1)
df_train.shape
def make_generator_model():

    model = tf.keras.Sequential()

    model.add(layers.Dense(16, use_bias=False, input_shape=(100,)))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Dense(16))

    assert model.output_shape == (None,16 ) # Note: None is the batch size

    

    model.add(layers.Dense(32))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

    

    model.add(layers.Dense(32))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

    

    model.add(layers.Dense(8))

    assert model.output_shape == (None,8 )

   

    

    return model
def make_discriminator_model():

    model = tf.keras.Sequential()

    model.add(layers.Dense(16, use_bias=False,

                                    input_shape=[1,8]))

   

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))



    model.add(layers.Dense(32, use_bias=True))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))



   

    model.add(layers.Dense(1))

   # model.add(layers.Softmax())



    return model
generator=make_generator_model()

discriminator=make_discriminator_model()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)

    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    total_loss = real_loss + fake_loss

    return total_loss
bce = tf.keras.losses.BinaryCrossentropy()

loss = bce([1., 1., 1., 1.], [1., 1., 1., 1.])

print('Loss: ', loss.numpy())  # Loss: 11.522857
def generator_loss(fake_output):

    return cross_entropy(tf.ones_like(fake_output), fake_output)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)

discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
EPOCHS = 1000

noise_dim = 100

#num_examples_to_generate = 16

BATCH_SIZE = 64



@tf.function

def train_step(images):

    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    images=tf.reshape(images,(1,8))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

      generated_images = generator(noise, training=True)



      real_output = discriminator(images, training=True)

      fake_output = discriminator(generated_images, training=True)



      gen_loss = generator_loss(fake_output)

      disc_loss = discriminator_loss(real_output, fake_output)

     

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)



    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    

    return (gen_loss,disc_loss)
history=dict()

history['gen']=[]

history['dis']=[]

def train(dataset, epochs):

    for epoch in range(epochs):

        start = time.time()



        for batch in dataset:



           gen_loss,dis_loss= train_step(batch)

        history['gen'].append(gen_loss)

        history['dis'].append(dis_loss)

        print ('Time for epoch {} is {} sec '.format(epoch + 1, time.time()-start))
x_train=df_train.values

train(x_train,EPOCHS) 
plt.plot(history['gen'])
plt.plot(history['dis'])
y_test=df.label

x_test=df.drop('label',axis=1).values.reshape(-1,1,8)
y_pred=discriminator.predict(x_test)
from sklearn.metrics import average_precision_score, accuracy_score ,recall_score, f1_score



#range of output value

plt.figure(figsize=(7,7))

plt.plot(range(len(y_pred)),y_pred.reshape(-1,1))

plt.ylabel('y_pred')
#looking for the optimal probability threshold

#prob=[0.5,0.55,0.6,0.65,0.75,0.8,0.85,0.9,0.95,1,2,2.5,5]

prob=[-0.2,-0.1,-0.05,0,0.005,0.1]

for p in prob:

    pred_value =[1 if i<p   else 0 for i in y_pred]

    f1=f1_score(y_test,pred_value)

    acc=accuracy_score(y_test,pred_value)

    precision=average_precision_score(y_test,pred_value)

    recall= recall_score(y_test,pred_value)

    print(f'prob ={p} and f1 score ={f1} : accuracy={acc} : precision={precision} : recall={recall}')

    print(pred_value.count(0),pred_value.count(1))