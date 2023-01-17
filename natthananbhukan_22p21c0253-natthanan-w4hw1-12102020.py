import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import seaborn as sns

import os

import cv2

from kaggle_datasets import KaggleDatasets

import tensorflow as tf



import warnings

warnings.filterwarnings('ignore')



plt.style.use('fivethirtyeight')

plt.rcParams['figure.figsize'] = [16, 8]



print('Using Tensorflow version:', tf.__version__)
try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
TRAIN_PATH = '../input/thaimnist/train'

TEST_PATH = '../input/thaimnist/test'

GCS_DS_PATH = KaggleDatasets().get_gcs_path('thaimnist')
df_train_1digit = pd.read_csv('../input/thaimnist/mnist.train.map.csv')

df_train_more_digit = pd.read_csv('../input/thaimnist/train.rules.csv')
number_1digit_null = df_train_1digit.isnull().sum()

number_moredigit_null = df_train_more_digit.isnull().sum()



print("Number of null 1 digit\n",number_1digit_null, end="\n\n")

print("Number of null more than 1 digit\n",number_moredigit_null)
img = cv2.imread(os.path.join(TRAIN_PATH,df_train_1digit['id'][0]))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img.shape
df_train_1digit.head(10)
df_train_1digit['category'].value_counts().to_frame().plot(kind='bar')
df_train_more_digit.head(10)
df_train_more_digit['predict'].value_counts().to_frame().plot(kind='bar')
df_train_more_digit.head(10)
def show_train_img(df, cat):

    

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(24, 10))

    ten_random_samples = df[df['category'] == cat]['id'].sample(10).values

    

    for idx, image in enumerate(ten_random_samples):

        final_path = os.path.join(TRAIN_PATH,image)

        img = cv2.imread(final_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axes.ravel()[idx].imshow(img)

        axes.ravel()[idx].axis('off')

    plt.tight_layout()
show_train_img(df_train_1digit,0)
show_train_img(df_train_1digit,4)
show_train_img(df_train_1digit,5)
IMG_SIZE_h = 456

IMG_SIZE_w = 456

BATCH_SIZE = 32

LR = 0.0005

EPOCHS = 50

WARMUP = 10

AUTO = tf.data.experimental.AUTOTUNE
def joinPathTrain(path):

    new_path = os.path.join(GCS_DS_PATH,'train',path)

    return new_path
df_train_1digit['path'] = df_train_1digit['id'].apply(joinPathTrain)

df_train_1digit['category'] = df_train_1digit['category'].apply(float)
df_train_1digit.head(10)
drop_list = pd.read_csv('../input/drop-list-byp0002/drop_lists.csv')['id'].to_list()
df_train_1digit = df_train_1digit[~df_train_1digit['id'].isin(drop_list)]
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import KFold
KFoldNo = 5

kf = KFold(n_splits=KFoldNo,random_state=2020, shuffle=True)
KFold_df = []

for train_index, test_index in kf.split(df_train_1digit):

  KFold_df.append({"train": df_train_1digit.iloc[train_index],

                   "val": df_train_1digit.iloc[test_index]

                   }

                  )
def decode_image(filename, label=None, image_size=(IMG_SIZE_h, IMG_SIZE_w) , file =True):

    if file:

        filename = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(filename, channels=3)

    image = tf.image.resize(image, image_size)

    image = tf.image.rgb_to_grayscale(image)

    image = tf.cast(image,tf.float32) / 255.0

    if label is None:

        return image

    else:

        return image, label
dataset_KFOLD = []

for k in range(KFoldNo):

    train_paths = np.array(KFold_df[k]["train"]['path'].to_list())

    train_labels = np.array(KFold_df[k]["train"]['category'].to_list())

    train_labels = to_categorical(train_labels)

    

    valid_paths = np.array(KFold_df[k]["val"]['path'].to_list())

    valid_labels = np.array(KFold_df[k]["val"]['category'].to_list())

    valid_labels = to_categorical(valid_labels)



    train_dataset = (

        tf.data.Dataset

        .from_tensor_slices((train_paths, train_labels))

        .map(decode_image, num_parallel_calls=AUTO)

        .cache()

        .repeat()

        .shuffle(2048)

        .batch(BATCH_SIZE)

        .prefetch(AUTO)

    )



    val_dataset = (

        tf.data.Dataset

        .from_tensor_slices((valid_paths, valid_labels))

        .map(decode_image, num_parallel_calls=AUTO)

        .batch(BATCH_SIZE)

        .cache()

        .prefetch(AUTO)

    )

    

    dataset_KFOLD.append({

        "train":train_dataset,

        "val":val_dataset

    })
from tensorflow.keras.optimizers import RMSprop,Adam



optimizer_RMS = RMSprop(lr=LR, rho=0.9, epsilon=1e-08, decay=0.0)

optimizer_Adam = Adam(lr=LR)
import keras.backend as K



def categorical_focal_loss_with_label_smoothing(gamma=2.0, alpha=0.25, ls=0.1, classes=10.0):

    """

    Implementation of Focal Loss from the paper in multiclass classification

    Formula:

        loss = -alpha*((1-p)^gamma)*log(p)

        y_ls = (1 - α) * y_hot + α / classes

    Parameters:

        alpha -- the same as wighting factor in balanced cross entropy

        gamma -- focusing parameter for modulating factor (1-p)

        ls    -- label smoothing parameter(alpha)

        classes     -- No. of classes

    Default value:

        gamma -- 2.0 as mentioned in the paper

        alpha -- 0.25 as mentioned in the paper

        ls    -- 0.1

        classes -- 10

    """

    def focal_loss(y_true, y_pred):

        # Define epsilon so that the backpropagation will not result in NaN

        # for 0 divisor case

        epsilon = K.epsilon()

        # Add the epsilon to prediction value

        #y_pred = y_pred + epsilon

        #label smoothing

        y_pred_ls = (1 - ls) * y_pred + ls / classes

        # Clip the prediction value

        y_pred_ls = K.clip(y_pred_ls, epsilon, 1.0-epsilon)

        # Calculate cross entropy

        cross_entropy = -y_true*K.log(y_pred_ls)

        # Calculate weight that consists of  modulating factor and weighting factor

        weight = alpha * y_true * K.pow((1-y_pred_ls), gamma)

        # Calculate focal loss

        loss = weight * cross_entropy

        # Sum the losses in mini_batch

        loss = K.sum(loss, axis=1)

        return loss

    

    return focal_loss
!pip install -q efficientnet
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, Input

from efficientnet.tfkeras import EfficientNetB3
model_KFOLD = []

for k in range(KFoldNo):

    with strategy.scope():

        model = tf.keras.Sequential([

        Input(shape=(IMG_SIZE_h,IMG_SIZE_w,1),name="Input"),

        Conv2D(3,1,name="ConvLayerInput"),

        EfficientNetB3(weights='noisy-student',

                       include_top=False,

                       pooling='avg'),



            Dense(10, activation='softmax')

        ])



        model.layers[0].trainable = False



        model.compile(optimizer = 'adam',

                      loss = categorical_focal_loss_with_label_smoothing(gamma=2.0, alpha=0.75, ls=0.125, classes=10.0), # num classes

                      metrics=['accuracy'])



        model_KFOLD.append(model)
model_KFOLD[0].summary()
from tensorflow.keras.callbacks import EarlyStopping

import math



def get_cosine_schedule_with_warmup(lr, num_warmup_steps, num_training_steps, num_cycles=0.5):

    def lrfn(epoch):

        if epoch < num_warmup_steps:

            return float(epoch) / float(max(1, num_warmup_steps)) * lr

        progress = float(epoch - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr



    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)



lr_schedule= get_cosine_schedule_with_warmup(lr=LR, num_warmup_steps=WARMUP, num_training_steps=EPOCHS)





es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
history_KFOLD = []

for k in range(KFoldNo):

    print("#"*50)

    print("FOLD",k)

    print("#"*50)

    history = model_KFOLD[k].fit(dataset_KFOLD[k]["train"],

                    epochs = EPOCHS, 

                    validation_data = dataset_KFOLD[k]["val"],

                    verbose = 1, 

                    steps_per_epoch=KFold_df[0]["train"].shape[0] // BATCH_SIZE, 

                    callbacks=[lr_schedule,es])
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
from sklearn.metrics import confusion_matrix





def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i in range (cm.shape[0]):

        for j in range (cm.shape[1]):

            plt.text(j, i, cm[i, j],

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predict the values from the validation dataset

Y_pred = model_KFOLD[0].predict(dataset_KFOLD[0]["val"])

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors'

valid_labels = np.array(KFold_df[0]["val"]['category'].to_list())

valid_labels = to_categorical(valid_labels)

Y_true = np.argmax(valid_labels,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
df_train_1digit = pd.read_csv('../input/thaimnist/mnist.train.map.csv')
df_train_mse = df_train_more_digit.copy()

df_train_mse_model = df_train_more_digit.copy()
df_train_mse.isnull().sum()
id_train_list_id = df_train_1digit.loc[:,['id','category']]['id'].to_list()

id_train_list_cat = df_train_1digit.loc[:,['id','category']]['category'].to_list()
dict_map = {}

for i in zip(id_train_list_id,id_train_list_cat):

    dict_map[i[0]] = i[1]
df_train_mse['feature1'] = df_train_mse['feature1'].map(dict_map)

df_train_mse['feature2'] = df_train_mse['feature2'].map(dict_map)

df_train_mse['feature3'] = df_train_mse['feature3'].map(dict_map)



df_train_mse_model['feature1'] = df_train_mse_model['feature1'].map(dict_map)

df_train_mse_model['feature2'] = df_train_mse_model['feature2'].map(dict_map)

df_train_mse_model['feature3'] = df_train_mse_model['feature3'].map(dict_map)
df_train_mse.isnull().sum()
from sklearn.metrics import mean_absolute_error as MAE
F2 = df_train_mse[df_train_mse['feature1'].isnull()]['feature2']

F3 = df_train_mse[df_train_mse['feature1'].isnull()]['feature3'] 

y_true_rule = df_train_mse[df_train_mse['feature1'].isnull()]['predict'].to_list()
y_pred_rule = F2 + F3
MAE(y_true_rule,y_pred_rule.to_list())
F2 = df_train_mse[df_train_mse['feature1'] == 0]['feature2']

F3 = df_train_mse[df_train_mse['feature1'] == 0]['feature3'] 

y_true_rule = df_train_mse[df_train_mse['feature1'] == 0]['predict'].to_list()
y_pred_rule = F2 * F3
MAE(y_true_rule,y_pred_rule.to_list())
F2 = df_train_mse[df_train_mse['feature1'] == 1]['feature2']

F3 = df_train_mse[df_train_mse['feature1'] == 1]['feature3'] 

y_true_rule = df_train_mse[df_train_mse['feature1'] == 1]['predict'].to_list()
y_pred_rule = np.abs(F2 - F3)
MAE(y_true_rule,y_pred_rule.to_list())
F2 = df_train_mse[df_train_mse['feature1'] == 2]['feature2']

F3 = df_train_mse[df_train_mse['feature1'] == 2]['feature3']

y_true_rule = df_train_mse[df_train_mse['feature1'] == 2]['predict'].to_list()
y_pred_rule = (F2 + F3) * np.abs(F2 - F3)
MAE(y_true_rule,y_pred_rule.to_list())
F2 = df_train_mse[df_train_mse['feature1'] == 3]['feature2']

F3 = df_train_mse[df_train_mse['feature1'] == 3]['feature3']

y_true_rule = df_train_mse[df_train_mse['feature1'] == 3]['predict'].to_list()
y_pred_rule = np.abs(((F3*(F3+1)) - (F2*(F2-1)))/2)
MAE(y_true_rule,y_pred_rule.to_list())
F2 = df_train_mse[df_train_mse['feature1'] == 4]['feature2']

F3 = df_train_mse[df_train_mse['feature1'] == 4]['feature3']

y_true_rule = df_train_mse[df_train_mse['feature1'] == 4]['predict'].to_list()
y_pred_rule = 50 + (F2-F3)
MAE(y_true_rule,y_pred_rule.to_list())
F2 = df_train_mse[df_train_mse['feature1'] == 5]['feature2']

F3 = df_train_mse[df_train_mse['feature1'] == 5]['feature3']

y_true_rule = df_train_mse[df_train_mse['feature1'] == 5]['predict'].to_list()
y_pred_rule = np.minimum(F2,F3)
MAE(y_true_rule,y_pred_rule.to_list())
F2 = df_train_mse[df_train_mse['feature1'] == 6]['feature2']

F3 = df_train_mse[df_train_mse['feature1'] == 6]['feature3']

y_true_rule = df_train_mse[df_train_mse['feature1'] == 6]['predict'].to_list()
y_pred_rule = np.maximum(F2,F3)
MAE(y_true_rule,y_pred_rule.to_list())
F2 = df_train_mse[df_train_mse['feature1'] == 7]['feature2']

F3 = df_train_mse[df_train_mse['feature1'] == 7]['feature3']

y_true_rule = df_train_mse[df_train_mse['feature1'] == 7]['predict'].to_list()
y_pred_rule = ((F2 * F3) % 9) * 11
MAE(y_true_rule,y_pred_rule.to_list())
F2 = df_train_mse[df_train_mse['feature1'] == 8]['feature2']

F3 = df_train_mse[df_train_mse['feature1'] == 8]['feature3']

y_true_rule = df_train_mse[df_train_mse['feature1'] == 8]['predict'].to_list()
y_pred_rule = ((F2**2)+1)*(F2) + (F3)*(F3+1)
def moreThan99(value):

    if value > 99:

        return value % 99

    else:

        return value
y_pred_rule = y_pred_rule.apply(moreThan99)
MAE(y_true_rule,y_pred_rule.to_list())
F2 = df_train_mse[df_train_mse['feature1'] == 9]['feature2']

F3 = df_train_mse[df_train_mse['feature1'] == 9]['feature3']

y_true_rule = df_train_mse[df_train_mse['feature1'] == 9]['predict'].to_list()
y_pred_rule = 50 + F2
MAE(y_true_rule,y_pred_rule.to_list())
def ruleBaseModel(df):

    F1_list = df.loc[:,['feature1']]['feature1'].to_numpy()

    F2_list = df.loc[:,['feature2']]['feature2'].to_numpy()

    F3_list = df.loc[:,['feature3']]['feature3'].to_numpy()

    y_pred = np.zeros([len(F1_list)])

    c = 0

    for value in zip(F1_list,F2_list,F3_list):

        F1 = value[0]

        F2 = value[1]

        F3 = value[2]

        if F1 != F1: # F1 == NaN

            y_pred[c] = F2 + F3

        elif F1 == 0:

            y_pred[c] = F2 * F3

        elif F1 == 1:

            y_pred[c] = np.abs(F2 - F3)

        elif F1 == 2:

            y_pred[c] = (F2 + F3)*np.abs(F2 - F3)

        elif F1 == 3:

            y_pred[c] = np.abs(((F3*(F3+1)) - (F2*(F2-1)))/2)

        elif F1 == 4:

            y_pred[c] = 50 + (F2 - F3)

        elif F1 == 5:

            y_pred[c] = np.minimum(F2,F3)

        elif F1 == 6:

            y_pred[c] = np.maximum(F2,F3)

        elif F1 == 7:

            y_pred[c] = ((F2 * F3) % 9) * 11

        elif F1 == 8:

            temp = ((F2**2)+1)*(F2) + (F3)*(F3+1)

            y_pred[c] = moreThan99(temp)

        elif F1 == 9:

            y_pred[c] = 50 + F2

        c+=1

    return y_pred
y_pred = ruleBaseModel(df_train_mse)
y_true = df_train_mse['predict'].to_list()
MAE(y_true,y_pred)
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV
df_train_mse_model.isnull().sum()
np.random.seed(0)

nan_rows = df_train_mse_model['feature1'].isna()

random_age = np.random.choice(df_train_mse_model['feature1'][~nan_rows], replace=True, size=sum(nan_rows))

df_train_mse_model.loc[nan_rows,'feature1'] = random_age
X_train_value = df_train_mse_model.loc[:,['feature1','feature2','feature3']].to_numpy()

Y_train_value = df_train_mse_model.loc[:,['predict']].to_numpy()
X_train_value
Y_train_value
X_train, X_test, y_train, y_test = train_test_split(X_train_value, Y_train_value, test_size=0.1, random_state=42)
regressor = DecisionTreeRegressor(random_state=2020,)
regressor.fit(X_train,y_train)
regressor.score(X_train,y_train)
y_pred = regressor.predict(X_test)
MAE(y_test,y_pred)
test_more_onedigit = pd.read_csv('../input/thaimnist/test.rules.csv')

df_test = test_more_onedigit.copy()
def joinPathTest(path):

    new_path = os.path.join(GCS_DS_PATH,'test',path)

    return new_path
df_test.isnull().sum()
df_test.loc[~df_test['feature1'].isnull(),'feature1'] = df_test.loc[~df_test['feature1'].isnull(),'feature1'].apply(joinPathTest)

df_test['feature2'] = df_test['feature2'].apply(joinPathTest)

df_test['feature3'] = df_test['feature3'].apply(joinPathTest)
df_test
feature1_list = df_test.loc[~df_test['feature1'].isnull(),'feature1'].to_list()

feature2_list = df_test['feature2'].to_list()

feature3_list = df_test['feature3'].to_list()
test_feature1_dataset = (

    tf.data.Dataset

    .from_tensor_slices(feature1_list)

    .map(decode_image, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

    .cache()

)



test_feature2_dataset = (

    tf.data.Dataset

    .from_tensor_slices(feature2_list)

    .map(decode_image, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

    .cache()

)



test_feature3_dataset = (

    tf.data.Dataset

    .from_tensor_slices(feature3_list)

    .map(decode_image, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

    .cache()

)
Feature1_KFold_pred = []

for k in range(KFoldNo):

  pred_feature1 = model_KFOLD[k].predict(test_feature1_dataset, verbose=1)

  pred_feature1 = pred_feature1.argmax(axis=1)

  Feature1_KFold_pred.append(pred_feature1)

Feature1_KFold_pred = np.array(Feature1_KFold_pred)
Feature2_KFold_pred = []

for k in range(KFoldNo):

  pred_feature2 = model_KFOLD[k].predict(test_feature2_dataset, verbose=1)

  pred_feature2 = pred_feature2.argmax(axis=1)

  Feature2_KFold_pred.append(pred_feature2)

Feature2_KFold_pred = np.array(Feature2_KFold_pred)
Feature3_KFold_pred = []

for k in range(KFoldNo):

  pred_feature3 = model_KFOLD[k].predict(test_feature3_dataset, verbose=1)

  pred_feature3 = pred_feature3.argmax(axis=1)

  Feature3_KFold_pred.append(pred_feature3)

Feature3_KFold_pred = np.array(Feature3_KFold_pred)
from scipy.stats import mode
pred_feature1 = mode(Feature1_KFold_pred)[0]

pred_feature2 = mode(Feature2_KFold_pred)[0]

pred_feature3 = mode(Feature3_KFold_pred)[0]
df_test.loc[~df_test['feature1'].isnull(),'feature1'] = pred_feature1[0]

df_test['feature2'] = pred_feature2[0]

df_test['feature3'] = pred_feature3[0]
df_test
y_pred = ruleBaseModel(df_test)
df_test['predict'] = y_pred
df_test = df_test.loc[:,['id','predict']]
df_test = df_test.set_index('id')

df_test['predict'] = df_test['predict'].apply(int)
df_test
df_test.to_csv("submissions.csv")