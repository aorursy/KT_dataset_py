# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Ignore  the warnings

import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



# data visualisation and manipulation

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

 

#configure

# sets matplotlib to inline and displays graphs below the corressponding cell.

%matplotlib inline  

style.use('fivethirtyeight')

sns.set(style='whitegrid',color_codes=True)



#model selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder



#preprocess.

from keras.preprocessing.image import ImageDataGenerator



#dl libraraies

import keras

from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense , merge

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from keras.utils import to_categorical

from keras.utils.vis_utils import model_to_dot

from keras.callbacks import ReduceLROnPlateau





from keras.layers.merge import dot

from keras.models import Model





# specifically for deeplearning.

from keras.layers import Dropout, Flatten,Activation,Input,Embedding

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

import tensorflow as tf

import random as rn

from IPython.display import SVG

 

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.

import cv2                  

import numpy as np  

from tqdm import tqdm

import os                   

from random import shuffle  

from zipfile import ZipFile

from PIL import Image







from surprise import Reader, Dataset, SVD

from surprise.model_selection import cross_validate



import warnings; warnings.simplefilter('ignore')
df = pd.read_csv('/kaggle/input/news_articles.csv')
df.head(10)
#total articles

len(df['Article_Id'])
#unique aticles

list_of_articleid = []

q = df['Article_Id'].unique()

list_of_articleid = list_of_articleid.append(q)

list_of_articleid
import scipy

import random

from scipy import stats
random.seed(15)



user_session = stats.geom.rvs(size=4831,  # Generate geometric data

                                  p=0.3)       # With success prob 0.5
user_session.size,user_session.max(),user_session.min()
user_session[:10],sum(user_session)


count_dict = {x : list(user_session).count(x) for x in user_session}

count_dict
#depicts number of users per number of sessions

    

bins = np.arange(0, 10, 1) # fixed bin size



plt.xlim([min(user_session)-1, max(user_session) +1])



plt.hist(user_session, bins=bins, alpha=0.5)

plt.title("Count of Number of users per session")

plt.xlabel('Number of sessions (bin size = 1)')

plt.ylabel('count')



plt.show()
import numpy as np



user_Id = range(1,4831)
userId_session = list(zip(user_Id,[10*i for i in user_session]))
type(userId_session), userId_session[:5]
#Calculating total number of articles served in a day in all sessions (may be clicked or not)



sum1 = 0

for i in range(len(userId_session)):

    

    sum1 += userId_session[i][1]

    

sum1
UserIDs = []



for i in range(len(userId_session)):

    

    for j in range(userId_session[i][1]):

        UserIDs.append(userId_session[i][0])
len(UserIDs)   #matches with sum1 above
UserIDs[:20]   # UserIds generated for all sessions the user opens
session_list = list(user_session)

session_list[:10]
session_Id =[]



for i in session_list:

    

    for j in range(1,i+1):

#         print j

        session_Id.append([j for i in range(10)])
session_Id = np.array(session_Id).flatten()


session_Id.shape
User_session = list(zip(UserIDs,session_Id ))


len(User_session),type(User_session)
import pandas as pd



df = pd.DataFrame(User_session, columns=['UserId', 'SessionId'])
df.tail(20)
Article_Id = list(range(4831))
type(Article_Id)
161730/4831
Article_Id = Article_Id*int(161730/4831)  



len(Article_Id)
import random

for x in range(len(User_session)-len(Article_Id)):

    Article_Id.append(random.randint(1,4831))
len(Article_Id)
from random import shuffle

shuffle(Article_Id)
c = len(df['UserId'])
Article_Id = Article_Id[:c]
df['ArticleId_served'] = Article_Id
df.tail()
len(df['UserId'].unique())
df
p = len(df['UserId'])
import random

numLow = 1 

numHigh = 6

x = []

for i in range (0,p):

    m = random.sample(range(numLow, numHigh), 1)

    x.append(m)
x[:3]
flat_list = []

for sublist in x:

    for item in sublist:

        flat_list.append(item)
len(flat_list)
df.head()
df['rating'] = flat_list
len(df['rating'])
df.head()
# df.to_csv('file1.csv') 

# saving the dataframe 

df.to_csv('file3.csv', index=False)



index=list(df['UserId'].unique())

columns=list(df['ArticleId_served'].unique())

index=sorted(index)

columns=sorted(columns)

 

util_df=pd.pivot_table(data=df,values='rating',index='UserId',columns='ArticleId_served')
util_df
util_df.fillna(0)
# x_train,x_test,y_train,y_test=train_test_split(df[['UserId','ArticleId_served']],df[['rating']],test_size=0.20,random_state=42)

users = df.UserId.unique()

movies = df.ArticleId_served.unique()



userid2idx = {o:i for i,o in enumerate(users)}

movieid2idx = {o:i for i,o in enumerate(movies)}
users
df['ArticleId_served'].head(70)
df['UserId'] = df['UserId'].apply(lambda x: userid2idx[x])

df['ArticleId_served'] = df['ArticleId_served'].apply(lambda x: movieid2idx[x])

split = np.random.rand(len(df)) < 0.8

train = df[split]

valid = df[~split]

print(train.shape , valid.shape)
df['ArticleId_served'].head(70)
n_article=len(df['ArticleId_served'].unique())

n_users=len(df['UserId'].unique())

n_latent_factors=64  # hyperparamter to deal with. 
user_input=Input(shape=(1,),name='user_input',dtype='int64')
user_input.shape
# tf.keras.layers.Embedding(

#      input_dim,

#      output_dim,

#      embeddings_initializer="uniform",

#      embeddings_regularizer=None,

#      activity_regularizer=None,

#      embeddings_constraint=None,

#      mask_zero=False,

#      input_length=None,

#      **kwargs

#  )
user_embedding=Embedding(n_users,n_latent_factors,name='user_embedding')(user_input)

user_embedding.shape
user_vec =Flatten(name='FlattenUsers')(user_embedding)

user_vec.shape
article_input=Input(shape=(1,),name='article_input',dtype='int64')

article_embedding=Embedding(n_article,n_latent_factors,name='article_embedding')(article_input)

article_vec=Flatten(name='FlattenArticles')(article_embedding)

# article_vec
article_vec
sim=dot([user_vec,article_vec],name='Simalarity-Dot-Product',axes=1)

model =keras.models.Model([user_input, article_input],sim)

model.summary()
# Model.compile(

#     optimizer="rmsprop",

#     loss=None,

#     metrics=None,

#     loss_weights=None,

#     weighted_metrics=None,

#     run_eagerly=None,

#     **kwargs

# )

model.compile(optimizer=Adam(lr=1e-4),loss='mse')
train.shape
train.shape

batch_size=128

epochs=50


# Model.fit(

#     x=None,

#     y=None,

#     batch_size=None,

#     epochs=1,

#     verbose=1,

#     callbacks=None,

#     validation_split=0.0,

#     validation_data=None,

#     shuffle=True,

#     class_weight=None,

#     sample_weight=None,

#     initial_epoch=0,

#     steps_per_epoch=None,

#     validation_steps=None,

#     validation_batch_size=None,

#     validation_freq=1,

#     max_queue_size=10,

#     workers=1,

#     use_multiprocessing=False,

# )
History = model.fit([train.UserId,train.ArticleId_served],train.rating, batch_size=batch_size,

                              epochs =epochs, validation_data = ([valid.UserId,valid.ArticleId_served],valid.rating),

                              verbose = 1)
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5

import matplotlib.pyplot as plt

plt.plot(History.history['loss'] , 'g')

plt.plot(History.history['val_loss'] , 'b')

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.grid(True)

plt.show()