from IPython.display import Image

Image(filename='../input/landmark-retrieval-2020/train/0/0/0/000014b1f770f640.jpg') 



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import tensorflow as tf

import cv2

import skimage.io

from cv2 import imread as cv2_imread

from cv2 import resize as cv2_resize

from skimage.io import imread

from skimage.transform import resize

from imgaug import augmenters as iaa

from sklearn import preprocessing

from sklearn.preprocessing import LabelBinarizer,LabelEncoder

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers

from tensorflow.keras.layers import Input, Add, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,GlobalAveragePooling2D,Concatenate, ReLU, LeakyReLU,Reshape, Lambda

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.optimizers import Adam,SGD

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential, load_model, Model

from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.keras.preprocessing import image

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import metrics

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications import InceptionV3

from tensorflow.keras.initializers import glorot_uniform

from tqdm import tqdm

import imgaug as ia

from imgaug import augmenters as iaa

from PIL import Image

import keras.backend as K

K.set_image_data_format('channels_last')

K.set_learning_phase(1)





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())



tf.debugging.set_log_device_placement(True)



# Create some tensors

a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

c = tf.matmul(a, b)



print(c)
train = pd.read_csv("../input/landmark-retrieval-2020/train.csv")

def get_paths(sub):

    index = ["0","1","2","3","4","5","6","7","8","9","a","b","c","d","e","f"]



    paths = []



    for a in index:

        for b in index:

            for c in index:

                try:

                    paths.extend([f"../input/landmark-retrieval-2020/{sub}/{a}/{b}/{c}/" + x for x in os.listdir(f"../input/landmark-retrieval-2020/{sub}/{a}/{b}/{c}")])

                except:

                    pass



    return paths


train_path = train

train_path["id"] = train_path.id.map(lambda path: f"../input/landmark-retrieval-2020/train/{path[0]}/{path[1]}/{path[2]}/{path}.jpg")
##Old implementation - changed after suggestion from @nawidsayed

'''

train_path = train



rows = []

for i in tqdm(range(len(train))):

    row = train.iloc[i]

    path  = list(row["id"])[:3]

    temp = row["id"]

    row["id"] = f"../input/landmark-retrieval-2020/train/{path[0]}/{path[1]}/{path[2]}/{temp}.jpg"

    rows.append(row["id"])

    

rows = pd.DataFrame(rows)

train_path["id"] = rows

'''
batch_size = 4

seed = 42

IMAGE_SIZE = 128

shape = (IMAGE_SIZE, IMAGE_SIZE, 3) ##desired shape of the image for resizing purposes

val_sample = 0.2 # % of validation sample

train_labels = pd.read_csv('../input/landmark-retrieval-2020/train.csv')

train_labels.head()
k =train[['id','landmark_id']].groupby(['landmark_id']).agg({'id':'count'})

k.rename(columns={'id':'Count_class'}, inplace=True)

k.reset_index(level=(0), inplace=True)

freq_ct_df = pd.DataFrame(k)

freq_ct_df.head()
train_labels = pd.merge(train,freq_ct_df, on = ['landmark_id'], how='left')

train_labels.head()
freq_ct_df.sort_values(by=['Count_class'],ascending=False,inplace=True)

freq_ct_df.head()
TOP = 40000
freq_ct_df_top = freq_ct_df.iloc[:TOP]

top_class = freq_ct_df_top['landmark_id'].tolist()
topclass_train = train_path[train_path['landmark_id'].isin (top_class) ]

topclass_train.shape
def getTrainParams():

    print("Encoding labels")

    data = topclass_train.copy()

    le = preprocessing.LabelEncoder()

    print("fitting LabelENcoder")

    data['label'] = le.fit_transform(data['landmark_id'])

    print("Success in LabelENcoder")

    lbls = topclass_train['landmark_id'].tolist()

    lb = LabelBinarizer(sparse_output=False)

    print("fitting LabelBinarizer")

    labels = lb.fit_transform(lbls)

    print("Success in LabelBinarizer")

    

    x = np.array(topclass_train['id'].tolist())

    print("id to array")

    y = labels

#     print(y.nbytes)

    print("labels to array")

    return x,y,le
class Landmark2020_DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, paths, labels, batch_size, shape, shuffle = False, use_cache = False, augment = False):

        self.paths, self.labels = paths, labels

        self.batch_size = batch_size

        self.shape = shape

        self.shuffle = shuffle

        self.use_cache = use_cache

        self.augment = augment

        if use_cache == True:

            self.cache = np.zeros((paths.shape[0], shape[0], shape[1], shape[2]), dtype=np.float16)

            self.is_cached = np.zeros((paths.shape[0]))

        self.on_epoch_end()

    

    def __len__(self):

        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    

    def __getitem__(self, idx):

        indexes = self.indexes[idx * self.batch_size : (idx+1) * self.batch_size]



        paths = self.paths[indexes]

        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))

        # Generate data

        if self.use_cache == True:

            X = self.cache[indexes]

            for i, path in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):

                image = self.__load_image(path)

                self.is_cached[indexes[i]] = 1

                self.cache[indexes[i]] = image

                X[i] = image

        else:

            for i, path in enumerate(paths):

                X[i] = self.__load_image(path)



        y = self.labels[indexes]

                

        if self.augment == True:

            seq = iaa.Sequential([

                iaa.OneOf([

                    iaa.Fliplr(0.5), # horizontal flips

                    

                    iaa.ContrastNormalization((0.75, 1.5)),

                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),

                    iaa.Multiply((0.8, 1.2), per_channel=0.2),

                    

                    iaa.Affine(rotate=0),

                    #iaa.Affine(rotate=90),

                    #iaa.Affine(rotate=180),

                    #iaa.Affine(rotate=270),

                    iaa.Fliplr(0.5),

                    #iaa.Flipud(0.5),

                ])], random_order=True)



            X = np.concatenate((X, seq.augment_images(X), seq.augment_images(X), seq.augment_images(X)), 0)

            y = np.concatenate((y, y, y, y), 0)

        

        return X, y

    

    def on_epoch_end(self):

        

        # Updates indexes after each epoch

        self.indexes = np.arange(len(self.paths))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def __iter__(self):

        """Create a generator that iterate over the Sequence."""

        for item in (self[i] for i in range(len(self))):

            yield item

            

    def __load_image(self, path):

        image_norm = cv2_imread(path)/255.0

        im = cv2_resize(image_norm,(IMAGE_SIZE,IMAGE_SIZE))

        return im
nlabls = topclass_train['landmark_id'].nunique()
from tensorflow.keras.applications import InceptionV3
NN_branches = 20

DENSE_UNITS = 2048
base_model = InceptionV3(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), 

                       weights=None, include_top=False)

base_model.load_weights("../input/keraspretrainedmodel/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")



input_image = Input((IMAGE_SIZE, IMAGE_SIZE, 3), dtype = tf.uint8)

x = tf.cast(input_image, tf.float32)

x = tf.keras.applications.inception_v3.preprocess_input(x)



x = base_model(input_image)

x = GlobalMaxPooling2D()(x)

x = Flatten()(x)

x = Dense(DENSE_UNITS,activation='relu')(x)

x = tf.keras.layers.LayerNormalization()(x)



embedding_model = Model(inputs = input_image,outputs = x,name='embedding_model')

cat_out_layers = []

output = None





for i in range(1,NN_branches+1):

    cat = Dense(DENSE_UNITS//NN_branches,activation = 'relu',name = 'cat'+str(i))(embedding_model.output)

        

    if i != NN_branches:

        cat_out = Dense(nlabls//NN_branches, activation='softmax', name = 'cat'+str(i)+'_out', kernel_initializer = glorot_uniform(seed=0))(cat)

    else:

        cat_out = Dense(nlabls//NN_branches+nlabls%NN_branches, activation='softmax', name = 'cat'+str(i)+'_out', kernel_initializer = glorot_uniform(seed=0))(cat)



    cat_out_layers.append(cat_out)    

        #     if output == None:

#         output = cat_out

#     else:

#         output = Concatenate()([output,cat_out])



output = Concatenate()(cat_out_layers)

# cat1 = Dense(512,activation = 'relu')(embedding_model.output)

# cat2 = Dense(512,activation = 'relu')(embedding_model.output)



# cat1_out = Dense(nlabls/2, activation='softmax',  kernel_initializer = glorot_uniform(seed=0))(cat1)

# cat2_out = Dense(nlabls/2, activation='softmax',  kernel_initializer = glorot_uniform(seed=0))(cat2)



# output = Concatenate()([cat1_out,cat2_out])

# output = Dense(nlabls, activation='softmax', name='fc' + str(nlabls), kernel_initializer = glorot_uniform(seed=0))(embedding_model.output)



model = Model(inputs=[input_image], outputs=[output])

model.summary()
from tensorflow.keras.utils import plot_model



plot_model(model,show_shapes=True)
embedding_model.summary()
from tensorflow.keras.metrics import categorical_accuracy,top_k_categorical_accuracy

def top_5_accuracy(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=5)
for layer in base_model.layers:

    layer.trainable = False
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc',top_5_accuracy])

model.summary()
import gc

gc.collect()
paths, labels,_ = getTrainParams()
import gc

gc.collect()
keys = np.arange(paths.shape[0], dtype=np.int)  

np.random.seed(seed)

np.random.shuffle(keys)

lastTrainIndex = int((1-val_sample) * paths.shape[0])



pathsTrain = paths[0:lastTrainIndex]

labelsTrain = labels[0:lastTrainIndex]



pathsVal = paths[lastTrainIndex:]

labelsVal = labels[lastTrainIndex:]



print(paths.shape, labels.shape)

print(pathsTrain.shape, labelsTrain.shape, pathsVal.shape, labelsVal.shape)
train_generator = Landmark2020_DataGenerator(pathsTrain, labelsTrain, batch_size, shape, use_cache=False, augment = False, shuffle = True)

val_generator = Landmark2020_DataGenerator(pathsVal, labelsVal, batch_size, shape, use_cache=False, shuffle = False)
epochs = 1

use_multiprocessing = True 

#workers = 1 
model.fit_generator(

    train_generator,

    steps_per_epoch=labelsTrain.shape[0]/batch_size,

    validation_data=val_generator,

    validation_steps=labelsVal.shape[0]/batch_size,

    #class_weight = class_weights,

    epochs=epochs,

    #callbacks = [clr],

    use_multiprocessing=use_multiprocessing,

    #workers=workers,

    verbose=1)


gc.collect()
from tensorflow.keras.utils import plot_model



plot_model(embedding_model,show_shapes=True)


gc.collect()
test_img = skimage.io.imread('../input/landmark-retrieval-2020/train/0/0/0/000014b1f770f640.jpg')

print(test_img.shape)

print(type(test_img))

y = model.predict(test_img.reshape(1,test_img.shape[0],test_img.shape[1],3))

y
class MyModel(tf.keras.Model):

    def __init__(self):

        super(MyModel, self).__init__()

        self.model = embedding_model

    

    @tf.function(input_signature=[

      tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8, name='input_image')

    ])

    

    def call(self, input_image):

        output_tensors = {}

        

        im = tf.image.resize(input_image, (IMAGE_SIZE, IMAGE_SIZE))

        

        extracted_features = self.model(tf.convert_to_tensor([im], dtype=tf.uint8))[0]

        output_tensors['global_descriptor'] = tf.identity(extracted_features, name='global_descriptor')

        return output_tensors
m = MyModel()



served_function = m.call



tf.saved_model.save(

    m, 

    export_dir="./model", 

    signatures={'serving_default': served_function}

)
from zipfile import ZipFile



with ZipFile('submission.zip','w') as output_zip_file:

    for filename in os.listdir('./model'):

        if os.path.isfile('./model/'+filename):

            output_zip_file.write('./model/'+filename, arcname=filename) 

    

    for filename in os.listdir('./model/variables'):

        if os.path.isfile('./model/variables/'+filename):

            output_zip_file.write('./model/variables/'+filename, arcname='variables/'+filename)

    

    for filename in os.listdir('./model/assets'):

        if os.path.isfile('./model/assets/'+filename):

            output_zip_file.write('./model/assets/'+filename, arcname='assets/'+filename)
!saved_model_cli show --dir ./model/ --all
image_ids = pd.read_csv(

    '../input/landmark-retrieval-2020/train.csv',

    nrows=100

)
def get_image(img_id):    

    chars = [char for char in img_id]

    dir_1, dir_2, dir_3 = chars[0], chars[1], chars[2]

    

    image = Image.open('../input/landmark-retrieval-2020/train/' + dir_1 + '/' + dir_2 + '/' + dir_3 + '/' + img_id + '.jpg')

    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image = np.asarray(image) / 255.0

    

    return image
images = [get_image(img_id) for img_id in image_ids.id]

images = np.array(images)
embeddings = embedding_model.predict(images)
from scipy.spatial import distance

distances = distance.cdist(embeddings, embeddings, 'euclidean')

print(distances.shape)
# ## We have removed the sensitive portions of this script, and included those

# ## that show you how we:

# ## 1. Load your model

# ## 2. Create embeddings

# ## 3. Compare and score those embeddings.

# ##

# ## Note that this means this code will NOT run as-is.



# import os

# import numpy as np

# from pathlib import Path

# import tensorflow as tf

# from PIL import Image

# import time

# from scipy.spatial import distance



# import solution

# import metrics



# REQUIRED_SIGNATURE = 'serving_default'

# REQUIRED_OUTPUT = 'global_descriptor'



# DATASET_DIR = '' # path to internal dataset



# SAVED_MODELS_DIR = os.path.join('kaggle', 'input')

# QUERY_IMAGE_DIR = os.path.join(DATASET_DIR, 'test')

# INDEX_IMAGE_DIR = os.path.join(DATASET_DIR, 'index')

# SOLUTION_PATH = ''



# def to_hex(image_id: int) -> str:

#     return '{0:0{1}x}'.format(image_id, 16)





# def show_elapsed_time(start):

#     hours, rem = divmod(time.time() - start, 3600)

#     minutes, seconds = divmod(rem, 60)

#     parts = []



#     if hours > 0:

#         parts.append('{:>02}h'.format(hours))



#     if minutes > 0:

#         parts.append('{:>02}m'.format(minutes))



#     parts.append('{:>05.2f}s'.format(seconds))



#     print('Elapsed Time: {}'.format(' '.join(parts)))





# def get_distance(scored_prediction):

#     return scored_prediction[1]



# embedding_fn = None



# def get_embedding(image_path: Path) -> np.ndarray:

#     image_data = np.array(Image.open(str(image_path)).convert('RGB'))

#     image_tensor = tf.convert_to_tensor(image_data)

#     return embedding_fn(image_tensor)[REQUIRED_OUTPUT].numpy()





# class Submission:

#     def __init__(self, name, model):

#         self.name = name

#         self.model = model

#         public_solution, private_solution, ignored_ids = solution.load(SOLUTION_PATH, 

#                                                          solution.RETRIEVAL_TASK_ID)

#         predictions = self.get_predictions()

        

#         self.private_score = self.get_metrics(predictions, private_solution)

#         self.public_score = self.get_metrics(predictions, public_solution)



#     def load(self, saved_model_proto_filename):

#         saved_model_path = Path(saved_model_proto_filename).parent

        

#         print (saved_model_path, saved_model_proto_filename)

        

#         name = saved_model_path.relative_to(SAVED_MODELS_DIR)

        

#         model = tf.saved_model.load(str(saved_model_path))

        

#         found_signatures = list(model.signatures.keys())

        

#         if REQUIRED_SIGNATURE not in found_signatures:

#             return None

        

#         outputs = model.signatures[REQUIRED_SIGNATURE].structured_outputs

#         if REQUIRED_OUTPUT not in outputs:

#             return None

        

#         global embedding_fn

#         embedding_fn = model.signatures[REQUIRED_SIGNATURE]



#         return Submission(name, model)

    



#     def get_id(self, image_path: Path):

#         return int(image_path.name.split('.')[0], 16)





#     def get_embeddings(self, image_root_dir: str):

#         image_paths = [p for p in Path(image_root_dir).rglob('*.jpg')]

        

#         embeddings = [get_embedding(image_path) 

#                       for i, image_path in enumerate(image_paths)]

#         ids = [self.get_id(image_path) for image_path in image_paths]



#         return ids, embeddings

    

#     def get_predictions(self):

#         print('Embedding queries...')

#         start = time.time()

#         query_ids, query_embeddings = self.get_embeddings(QUERY_IMAGE_DIR)

#         show_elapsed_time(start)



#         print('Embedding index...')

#         start = time.time()

#         index_ids, index_embeddings = self.get_embeddings(INDEX_IMAGE_DIR)

#         show_elapsed_time(start)



#         print('Computing distances...', end='\t')

#         start = time.time()

#         distances = distance.cdist(np.array(query_embeddings), 

#                                    np.array(index_embeddings), 'euclidean')

#         show_elapsed_time(start)



#         print('Finding NN indices...', end='\t')

#         start = time.time()

#         predicted_positions = np.argpartition(distances, K, axis=1)[:, :K]

#         show_elapsed_time(start)



#         print('Converting to dict...', end='\t')

#         predictions = {}

#         for i, query_id in enumerate(query_ids):

#             nearest = [(index_ids[j], distances[i, j]) 

#                        for j in predicted_positions[i]]

#             nearest.sort(key=lambda x: x[1])

#             prediction = [to_hex(index_id) for index_id, d in nearest]

#             predictions[to_hex(query_id)] = prediction

#         show_elapsed_time(start)



#         return predictions

    

#     def get_metrics(self, predictions, solution):

#         relevant_predictions = {}



#         for key in solution.keys():

#             if key in predictions:

#                 relevant_predictions[key] = predictions[key]



#         # Mean average precision.

#         mean_average_precision = metrics.MeanAveragePrecision(

#             relevant_predictions, solution, max_predictions=K)

#         print('Mean Average Precision (mAP): {:.4f}'.format(mean_average_precision))



#         return mean_average_precision

    

# ## after unpacking your zipped submission to /kaggle/working, the saved_model.pb

# ## file and attendant directory structure are passed to the the Submission object

# ## for loading.



# submission_object = Submission.load("/kaggle/working/saved_model.pb")