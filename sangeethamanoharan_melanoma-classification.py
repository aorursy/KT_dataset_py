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
!pip install -q efficientnet
!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.preprocessing import LabelEncoder





import numpy as np

import pandas as pd 

import os

import re

import cv2

import math

import time

from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import train_test_split

from kaggle_datasets import KaggleDatasets



import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Model,Sequential

from tensorflow.keras import optimizers

import efficientnet.tfkeras as efn



import seaborn as sns

import matplotlib.pyplot as plt

from plotly.subplots import make_subplots

import plotly.express as px
print("Tensorflow version " + tf.__version__)



AUTO = tf.data.experimental.AUTOTUNE



try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy()

     

print("REPLICAS: ", strategy.num_replicas_in_sync)



DATASET = '512x512-melanoma-tfrecords-70k-images'

GCS_PATH = KaggleDatasets().get_gcs_path(DATASET)
#Paths to train and test images

train_img_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'

test_img_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'
train = pd.DataFrame(pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv"))

test = pd.DataFrame(pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv"))
train.shape, test.shape
train.head()
test.head()
train.info()
train.isna().sum()
test.isna().sum()
train = train.dropna()



train.info()
test = test.dropna()



test.info()


train['patient_id'].nunique(), test["patient_id"].nunique()
print(train["target"].value_counts())
malignant = len(train[train["target"] == 1])

benign = len(train[train["target"] == 0])



labels = ["Malignant", "Benign"] 

size = [malignant, benign]



plt.figure(figsize = (8, 8))

plt.pie(size, labels = labels, shadow = True, startangle = 90, colors = ["r", "g"])

plt.title("Malignant VS Benign Cases")

plt.legend()
train_males = len(train[train["sex"] == "male"])

train_females  = len(train[train["sex"] == "female"])



test_males = len(test[test["sex"] == "male"])

test_females  = len(test[test["sex"] == "female"])

fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.countplot(x='sex',data=train,ax=ax[0])

ax[0].set_xlabel(" ")

ax[0].set_title("Gender counts in train set")



print("Number of males in training set = ", train_males)

print("Number of females in training set= ", train_females)

sns.countplot(x='sex',data=test,ax=ax[1])

ax[1].set_xlabel(" ")

ax[1].set_title("Gender counts in test set")

print("Number of males in testing set = ", test_males)

print("Number of females in testing set= ", test_females)

train_malignant  = train[train["target"] == 1]

train_malignant_males = len(train_malignant[train_malignant["sex"] == "male"])

train_malignant_females  = len(train_malignant[train_malignant["sex"] == "female"])



labels = ["Malignant Male Cases", "Malignant Female Cases"] 

size = [train_malignant_males, train_malignant_females]

explode = [0.1, 0.0]



plt.figure(figsize = (10, 10))

plt.pie(size, labels = labels, explode = explode, shadow = True, startangle = 90, colors = ["g", "b"])

plt.title("Malignant Male VS Female Cases", fontsize = 18)

plt.legend()

print("Malignant Male Cases = ", train_malignant_males)

print("Malignant Female Cases = ", train_malignant_females)
train_benign  = train[train["target"] == 0]



train_benign_males = len(train_benign[train_benign["sex"] == "male"])

train_benign_females  = len(train_benign[train_benign["sex"] == "female"]) 



labels = ["Benign Male Cases", "Benign Female Cases"] 

size = [train_benign_males, train_benign_females]

explode = [0.1, 0.0]



plt.figure(figsize = (10, 10))

plt.pie(size, labels = labels, explode = explode, shadow = True, startangle = 90, colors = ["g", "y"])

plt.title("Benign Male VS Benign Female Cases", fontsize = 18)

plt.legend()

print("Benign Male Cases = ", train_benign_males)

print("Benign Female Cases = ", train_benign_females)
cancer_versus_sex = train.groupby(["benign_malignant","sex"]).size()

print(cancer_versus_sex)
cancer_versus_sex = cancer_versus_sex.unstack(level = 1) / len(train) * 100

print(cancer_versus_sex)

print(type(cancer_versus_sex))
sns.set(style='whitegrid')

sns.set_context("paper", rc={"font.size":12,"axes.titlesize":20,"axes.labelsize":18})   



plt.figure(figsize = (10, 6))

sns.heatmap(cancer_versus_sex, annot=True, cmap="icefire", cbar=True)

plt.title("Cancer VS Sex Heatmap Analysis Normalized", fontsize = 18)

plt.tight_layout()
sns.set(style="whitegrid")

sns.set_context("paper", rc={"font_size":12,"axes.titlesize":20,"axes.labelsize":18})   

plt.figure(figsize = (10, 6))

sns.boxplot(train["benign_malignant"], train["age_approx"], palette="icefire")

plt.title("Age VS Cancer Boxplot Analysis")

plt.tight_layout()
temp_train = train.anatom_site_general_challenge.value_counts().sort_values(ascending=False)

temp_test = test.anatom_site_general_challenge.value_counts().sort_values(ascending=False)

print("Anatom_Site valuecounts for train data",temp_train)

print("Anatom_Site valuecounts for test data",temp_test)





fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.barplot(x=temp_train.index.values, y=temp_train.values,ax=ax[0])

ax[0].set_xlabel(" ")

labels = ax[0].get_xticklabels()

ax[0].set_xticklabels(labels, rotation=90)

ax[0].set_title("Image location in train set")



sns.barplot(x=temp_test.index.values, y=temp_test.values,ax=ax[1])

ax[1].set_xlabel(" ")

labels = ax[1].get_xticklabels()

ax[1].set_xticklabels(labels, rotation=90)

ax[1].set_title("Image location in test set")
fig, ax =plt.subplots(1,2,figsize=(20,7))

sns.countplot(x="age_approx", data= train,ax=ax[0])

ax[0].set_xlabel("")

ax[0].set_title("Age distribution in train set")

sns.countplot(x="age_approx", data=test, ax=ax[1])

ax[0].set_xlabel("")

ax[1].set_title("Age distribution in test set")
train_ages_benign = train.loc[train["target"] == 0, "age_approx"]

train_ages_malignant = train.loc[train["target"] == 1 , "age_approx"]



plt.figure(figsize = (10, 8))

sns.kdeplot(train_ages_benign, label = "Benign", shade = True, legend = True, cbar = True)

sns.kdeplot(train_ages_malignant, label = "Malignant", shade = True, legend = True, cbar = True)

plt.grid(True)

plt.xlabel("Age Of The Patients", fontsize = 18)

plt.ylabel("Probability Density", fontsize = 18)

plt.grid(which = "minor", axis = "both")

plt.title("Probabilistic Age Distribution In Training Set", fontsize = 18)
print('No of samples:  ' + str(train.image_name.nunique()))

print('No of patients: ' + str(train.patient_id.nunique()))
image_freq_per_patient = train.groupby(['patient_id']).count()['image_name']

plt.hist(image_freq_per_patient.tolist(), bins = image_freq_per_patient.nunique())

plt.xlabel('No of samples per patient')

plt.ylabel('No of patients')

plt.show()

print('Minimum no of sample taken from  single patient', image_freq_per_patient.min())

print('Maximum no of sample taken from  single patient', image_freq_per_patient.max())

print('There are ',int( image_freq_per_patient.mean()), ' samples taken from each patients on average')

print('Median of no. of samples taken from  single patient', int(image_freq_per_patient.median()))

print('Mode of no. of samples taken from  single patient', int(image_freq_per_patient.mode()))
train.groupby(['benign_malignant', 'sex']).nunique()['patient_id']
category_sex = train.groupby(['sex', 'benign_malignant']).nunique()['patient_id'].tolist()



labels = ['Benign', 'Malignant']

benign_data = category_sex[0:2]

maglignant_data = category_sex[2:4]

x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars



fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, benign_data, width, label='Male')

rects2 = ax.bar(x + width/2, maglignant_data, width, label='Female')

ax.set_ylabel('No of patients')

ax.set_title('Patient Count by Benign and Malignant with Sex')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()



def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')

        

autolabel(rects1)

autolabel(rects2)



fig.tight_layout()

plt.show()
# === TRAIN ===

to_encode = ['sex', 'anatom_site_general_challenge', 'diagnosis']

encoded_all = []



label_encoder = LabelEncoder()



for column in to_encode:

    encoded = label_encoder.fit_transform(train[column])

    encoded_all.append(encoded)

    

train['sex'] = encoded_all[0]

train['anatom_site_general_challenge'] = encoded_all[1]

train['diagnosis'] = encoded_all[2]



if 'benign_malignant' in train.columns : train.drop(['benign_malignant'], axis=1, inplace=True)
# === TEST ===

to_encode = ['sex', 'anatom_site_general_challenge']

encoded_all = []



label_encoder = LabelEncoder()



for column in to_encode:

    encoded = label_encoder.fit_transform(test[column])

    encoded_all.append(encoded)

    

test['sex'] = encoded_all[0]

test['anatom_site_general_challenge'] = encoded_all[1]
train.head()
image_names = train["image_name"].values

image_names = image_names + ".jpg"

image_names
random_images = [np.random.choice(image_names) for i in range(4)] # Generates a random sample from a given 1-D array

random_images
train_dir = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"
plt.figure(figsize = (12, 8))

for i in range(4) : 

    plt.subplot(2, 2, i + 1) 

    image = cv2.imread(os.path.join(train_dir, random_images[i]))

    # cv2 reads images in BGR format. Hence we convert it to RGB

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image, cmap = "gray")

    plt.grid(True)

# Automatically adjust subplot parameters to give specified padding.

plt.tight_layout()
# Benign Cases

benign = train[train["target"] == 0] 

image_names = benign["image_name"].values

image_names = image_names + ".jpg"

benign_image_list = [np.random.choice(image_names) for i in tqdm(range(1000))]



red = []

green = [] 

blue = []



for image_name in tqdm(benign_image_list) : 

    image = cv2.imread(os.path.join(train_dir, image_name))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    

    mean_red = np.mean(image[:,:,0])

    mean_green = np.mean(image[:,:,1])

    mean_blue = np.mean(image[:,:,2])

    

    red.append(mean_red)

    green.append(mean_green)

    blue.append(mean_blue)
SEED = 42

BATCH_SIZE = 32 * strategy.num_replicas_in_sync

SIZE = [512,512]

LR = 0.00004

EPOCHS = 12

WARMUP = 5

WEIGHT_DECAY = 0

LABEL_SMOOTHING = 0.05

TTA = 4
def seed_everything(SEED):

    np.random.seed(SEED)

    tf.random.set_seed(SEED)



seed_everything(SEED)

train_filenames = tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')

test_filenames = tf.io.gfile.glob(GCS_PATH + '/test*.tfrec')
train_filenames,valid_filenames = train_test_split(train_filenames,test_size = 0.2,random_state = SEED)
def decode_image(image):

    image = tf.image.decode_jpeg(image, channels=3) 

    image = tf.cast(image, tf.float32)/255.0

    image = tf.reshape(image, [*SIZE, 3])

    return image



def data_augment(image, label=None, seed=SEED):

    image = tf.image.rot90(image,k=np.random.randint(4))

    image = tf.image.random_flip_left_right(image, seed=seed)

    image = tf.image.random_flip_up_down(image, seed=seed)

    if label is None:

        return image

    else:

        return image, label



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "target": tf.io.FixedLenFeature([], tf.int64),  }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['target'], tf.int32)

    return image, label 



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "image_name": tf.io.FixedLenFeature([], tf.string), }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    image_name = example['image_name']

    return image, image_name



def load_dataset(filenames, labeled=True, ordered=False):

    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False



    dataset = (tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) 

              .with_options(ignore_order)

              .map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO))

            

    return dataset



def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



def plot_transform(num_images):

    plt.figure(figsize=(30,10))

    x = load_dataset(train_filenames, labeled=False)

    image,_ = iter(x).next()

    for i in range(1,num_images+1):

        plt.subplot(1,num_images+1,i)

        plt.axis('off')

        image = data_augment(image=image)

        plt.imshow(image)
plot_transform(7)
train_dataset = (load_dataset(train_filenames, labeled=True)

    .map(data_augment, num_parallel_calls=AUTO)

    .shuffle(SEED)

    .batch(BATCH_SIZE,drop_remainder=True)

    .repeat()

    .prefetch(AUTO))



valid_dataset = (load_dataset(valid_filenames, labeled=True)

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO))
with strategy.scope():



    model = tf.keras.Sequential([

        efn.EfficientNetB7(input_shape=(*SIZE, 3),weights='imagenet',pooling='avg',include_top=False),

        Dense(1, activation='sigmoid')

    ])

    

    model.compile(

        optimizer='adam',

        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = LABEL_SMOOTHING),

        metrics=['accuracy',tf.keras.metrics.AUC(name='auc')])

    
model.summary()
def get_cosine_schedule_with_warmup(lr,num_warmup_steps, num_training_steps, num_cycles=0.5):

    def lrfn(epoch):

        if epoch < num_warmup_steps:

            return (float(epoch) / float(max(1, num_warmup_steps))) * lr

        progress = float(epoch - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr



    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)



lr_schedule= get_cosine_schedule_with_warmup(lr=LR,num_warmup_steps=WARMUP,num_training_steps=EPOCHS)
def train():

        STEPS_PER_EPOCH = count_data_items(train_filenames) // BATCH_SIZE

        history = model.fit(

            train_dataset, 

            epochs=EPOCHS, 

            callbacks=[lr_schedule],

            steps_per_epoch=STEPS_PER_EPOCH,

            validation_data=valid_dataset)



        string = 'Train acc:{:.4f} Train loss:{:.4f} AUC: {:.4f}, Val acc:{:.4f} Val loss:{:.4f} Val AUC: {:.4f}'.format( \

            model.history.history['accuracy'][-1],model.history.history['loss'][-1],\

            model.history.history['auc'][-1],\

            model.history.history['val_accuracy'][-1],model.history.history['val_loss'][-1],\

            model.history.history['val_auc'][-1])



        return string
train()
num_test_images = count_data_items(test_filenames)

submission_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')

for i in range(TTA):

    test_dataset = (load_dataset(test_filenames, labeled=False,ordered=True)

    .map(data_augment, num_parallel_calls=AUTO)  

    .batch(BATCH_SIZE))

    test_dataset_images = test_dataset.map(lambda image, image_name: image)

    test_dataset_image_name = test_dataset.map(lambda image, image_name: image_name).unbatch()

    test_ids = next(iter(test_dataset_image_name.batch(num_test_images))).numpy().astype('U')

    test_pred = model.predict(test_dataset_images, verbose=1) 

    pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(test_pred)})

    temp = submission_df.copy()   

    del temp['target']  

    submission_df['target'] += temp.merge(pred_df,on="image_name")['target']/TTA
submission_df.to_csv('submission.csv', index=False)

pd.Series(np.round(submission_df['target'].values)).value_counts() 