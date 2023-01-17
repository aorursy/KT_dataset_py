

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
os.listdir('../input/sc2datawithoutaug')
traindf_noisy=pd.read_csv('../input/freesound-audio-tagging-2019/train_noisy.csv',dtype=str)



traindf_curated=pd.read_csv('../input/freesound-audio-tagging-2019/train_curated.csv',dtype=str)
traindf_noisy.head()
temp_label=[]

temp_fname=[]



for file in os.listdir("../input/sc2datawithoutaug/Curated without AUG/Curated without AUG/curated_train"):

    temp_fname.append(file)

    if len(file.split("_"))==1:

         temp=file.split(".")[0]+".wav"

    elif len(file.split("_"))==2:  

        temp=file.split("_")[0]+".wav" 

    try:   

        label=traindf_noisy[traindf_noisy["fname"]==temp].iloc[0]["labels"]

    except IndexError:    

        label=traindf_curated[traindf_curated["fname"]==temp].iloc[0]["labels"]

    temp_label.append(label)
train_df = pd.DataFrame({'fname':temp_fname, 'labels':temp_label})
train_df.shape
temp_label=[]

temp_fname=[]



for file in os.listdir("../input/sc2datawithoutaug/Curated without AUG/Curated without AUG/curated_cv"):

    temp_fname.append(file)

    if len(file.split("_"))==1:

         temp=file.split(".")[0]+".wav"

    elif len(file.split("_"))==2:  

        temp=file.split("_")[0]+".wav" 

    try:   

        label=traindf_noisy[traindf_noisy["fname"]==temp].iloc[0]["labels"]

    except IndexError:    

        label=traindf_curated[traindf_curated["fname"]==temp].iloc[0]["labels"]

    temp_label.append(label)
cv_df = pd.DataFrame({'fname':temp_fname, 'labels':temp_label})

cv_df.shape
temp_label=[]

temp_fname=[]



for file in os.listdir("../input/sc2datawithoutaug/Curated without AUG/Curated without AUG/curated_test"):

    temp_fname.append(file)

    if len(file.split("_"))==1:

         temp=file.split(".")[0]+".wav"

    elif len(file.split("_"))==2:  

        temp=file.split("_")[0]+".wav" 

    try:   

        label=traindf_noisy[traindf_noisy["fname"]==temp].iloc[0]["labels"]

    except IndexError:    

        label=traindf_curated[traindf_curated["fname"]==temp].iloc[0]["labels"]

    temp_label.append(label)
    

test_df = pd.DataFrame({'fname':temp_fname, 'labels':temp_label})    

test_df.shape
from sklearn.preprocessing import MultiLabelBinarizer



mlb_train = MultiLabelBinarizer()





labels_train = mlb_train.fit_transform([ i.split(",") for i in list(train_df["labels"])])





labels_test = mlb_train.transform([ i.split(",") for i in list(test_df["labels"])])





#mlb_cv = MultiLabelBinarizer()

labels_cv = mlb_train.transform([ i.split(",") for i in list(cv_df["labels"])])

labels_test.shape
trainmultidf=pd.DataFrame(data=labels_train,columns=list(mlb_train.classes_))

trainmultidf["fname"]=list(train_df["fname"])



testmultidf=pd.DataFrame(data=labels_test,columns=list(mlb_train.classes_))

testmultidf["fname"]=list(test_df["fname"])





cvmultidf=pd.DataFrame(data=labels_cv,columns=list(mlb_train.classes_))

cvmultidf["fname"]=list(cv_df["fname"])

#We change the ids for the images in the csv files to reflect their new status as jpgs

#https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c

from keras_preprocessing.image import ImageDataGenerator







datagen=ImageDataGenerator(rescale=1./255.)





train_generator=datagen.flow_from_dataframe(

    dataframe=trainmultidf,

    directory="../input/sc2datawithoutaug/Curated without AUG/Curated without AUG/curated_train",

    x_col="fname",

    y_col=list(mlb_train.classes_),

    subset="training",

    batch_size=64,

    seed=42,

    shuffle=True,

    class_mode="raw",

    #color_mode="grayscale",

    target_size=(64,64))



train_generator.n
cvmultidf.shape
cvmultidf.head()


valid_datagen=ImageDataGenerator(rescale=1./255.)



valid_generator=valid_datagen.flow_from_dataframe(

    dataframe=cvmultidf,

    directory="../input/sc2datawithoutaug/Curated without AUG/Curated without AUG/curated_cv",

    x_col="fname",

    y_col=list(mlb_train.classes_),

   # subset="validation",

    batch_size=64,

    seed=42,

    shuffle=True,

    class_mode="raw",

    #color_mode="grayscale",

    target_size=(64,64))

cvmultidf.shape
test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(

    dataframe=testmultidf,

    directory="../input/sc2datawithoutaug/Curated without AUG/Curated without AUG/curated_test",

    x_col="fname",

    y_col=None,

    batch_size=64,

    seed=42,

    shuffle=False,

    class_mode=None,

   # color_mode="grayscale",

    target_size=(64,64))

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
from sklearn import model_selection

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.sequence import pad_sequences



from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.models import Model

from tensorflow.keras.models import load_model



import numpy as np

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing import sequence







import tensorflow as tf

from tensorflow.keras import models, layers

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

#from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping 

from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv2D, MaxPooling2D, Dropout, Activation, Input,BatchNormalization, AveragePooling2D,GlobalMaxPool2D,PReLU



from tensorflow.keras.models import model_from_json  

from tensorflow.keras.applications import DenseNet169



from tensorflow.keras.callbacks import (ModelCheckpoint, LearningRateScheduler,

                             EarlyStopping, ReduceLROnPlateau,CSVLogger)
#last_layer = model.get_layer('avg_pool').output



image_input = Input(shape=(64, 64, 3))

model = DenseNet169(input_tensor=image_input, include_top=True)

last_layer = model.get_layer('avg_pool').output

x= Flatten(name='flatten')(last_layer)

#model=





#output = Dense(80, activation='sigmoid', name='output_layer')(model.layers[-2].output)

x= Dense(80)(x)

output = Activation('sigmoid')(x)

#out = Dense(num_classes, activation='softmax', name='output_layer')(x)
model.layers[-4:]
model.layers[-4:]
custom_densenet169_model = Model(inputs=image_input,outputs= output)

custom_densenet169_model.summary()
custom_densenet169_model.load_weights("../input/sc2newfinalweights/NoisyTotal.best_weights_loss.hdf5")
#from tensorflow.keras.utils import plot_model

#plot_model(custom_densenet169_model, 'model_resnet50.png', show_shapes=True)
opt = tf.keras.optimizers.Adam(lr=0.0009)#tf.keras.optimizers.RMSprop(lr=0.3, decay=1e-6) 

#tf.keras.optimizers.Adam(lr=0.001)#RMSprop(lr=0.0001, decay=1e-6)



# Let's train the model using RMSprop

custom_densenet169_model.compile(loss=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM,label_smoothing=0.2),#label_smoothing=0.7#'categorical_crossentropy',

              optimizer=opt,

               metrics=['categorical_accuracy'])
#Fitting keras model, no test gen for now

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

#STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
STEP_SIZE_TRAIN
# simple early stopping

#earlyStop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100,)

#https://machinelearningmastery.com/check-point-deep-learning-models-keras/





#model_checkpoint = ModelCheckpoint('weights_cnn_lstm.best.hdf5', monitor='val_categorical_accuracy', mode='max', verbose=1, save_best_only=True)

#filepath="weights-improvement-{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5"





csv_logger = CSVLogger(filename='../working/training_log.csv',

                       separator=',',

                       append=True)

#https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.6,

                              patience=6, min_lr=0,verbose=1)





model_checkpoint = ModelCheckpoint("NoisyTotal.best_weights.hdf5", monitor='val_categorical_accuracy', mode='max', verbose=1, save_best_only=True)



# fit model



#es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, min_delta=0.001 )

es = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', mode='max', verbose=1, patience=30, min_delta=0.001 )



callbacks_list = [model_checkpoint, csv_logger, reduceLROnPlat,es]


#custom_densenet169_model.load_weights("../input/sc2weights/total.best_weights.hdf5")

#custom_densenet169_model.load_weights("../input/sc2weights/total.best_weights_iter2.hdf5")
history=custom_densenet169_model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=300,

                    callbacks=callbacks_list

)
#history.history
import matplotlib.pyplot as plt





loss_train = history.history['loss']

loss_val = history.history['val_loss']

#epochs = np.range(1,1)

plt.plot(loss_train, 'g', label='Training loss')

plt.plot(loss_val, 'b', label='validation loss')

plt.title('Training and Validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
loss_train = history.history['categorical_accuracy']

loss_val = history.history['val_categorical_accuracy']

epochs = range(1,41)

plt.plot(loss_train, 'g', label='Training accuracy')

plt.plot(loss_val, 'b', label='validation accuracy')

plt.title('Training and Validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
res = custom_densenet169_model.predict_generator(valid_generator, verbose=1)



    
test_generator.reset()

res_test=custom_densenet169_model.predict_generator(test_generator,

#steps=STEP_SIZE_TEST,

verbose=1)
valid_generator.reset()

res_cv=custom_densenet169_model.predict_generator(valid_generator,

#steps=STEP_SIZE_TEST,

verbose=1)
train_generator.reset()

res_train=custom_densenet169_model.predict_generator(train_generator,

#steps=STEP_SIZE_TEST,

verbose=1)
res_test.shape
# Converting taget and identity columns to booleans



target_columns=list(trainmultidf.columns)[:-1]



def convert_to_bool(df, col_name):

    df[col_name] = np.where(df[col_name] >= 0.5, True, False)

    

def convert_dataframe_to_bool(df):

    bool_df = df.copy()

    for col in target_columns:

        convert_to_bool(bool_df, col)

    return bool_df



test_bool = convert_dataframe_to_bool(testmultidf) 

test_lable_bool=test_bool[list(test_bool.columns)[:-1]].to_numpy()



train_bool = convert_dataframe_to_bool(trainmultidf) 

train_lable_bool=train_bool[list(train_bool.columns)[:-1]].to_numpy()



cv_bool = convert_dataframe_to_bool(cvmultidf) 

cv_lable_bool=cv_bool[list(cv_bool.columns)[:-1]].to_numpy()
cv_lable_bool.shape
import numpy as np

import sklearn.metrics
# Core calculation of label precisions for one test sample.



def _one_sample_positive_class_precisions(scores, truth):

  """Calculate precisions for each true class for a single sample.

  

  Args:

    scores: np.array of (num_classes,) giving the individual classifier scores.

    truth: np.array of (num_classes,) bools indicating which classes are true.



  Returns:

    pos_class_indices: np.array of indices of the true classes for this sample.

    pos_class_precisions: np.array of precisions corresponding to each of those

      classes.

  """

  num_classes = scores.shape[0]

  pos_class_indices = np.flatnonzero(truth > 0)

  # Only calculate precisions if there are some true classes.

  if not len(pos_class_indices):

    return pos_class_indices, np.zeros(0)

  # Retrieval list of classes for this sample. 

  retrieved_classes = np.argsort(scores)[::-1]

  # class_rankings[top_scoring_class_index] == 0 etc.

  class_rankings = np.zeros(num_classes, dtype=np.int)

  class_rankings[retrieved_classes] = range(num_classes)

  # Which of these is a true label?

  retrieved_class_true = np.zeros(num_classes, dtype=np.bool)

  retrieved_class_true[class_rankings[pos_class_indices]] = True

  # Num hits for every truncated retrieval list.

  retrieved_cumulative_hits = np.cumsum(retrieved_class_true)

  # Precision of retrieval list truncated at each hit, in order of pos_labels.

  precision_at_hits = (

      retrieved_cumulative_hits[class_rankings[pos_class_indices]] / 

      (1 + class_rankings[pos_class_indices].astype(np.float)))

  return pos_class_indices, precision_at_hits

# All-in-one calculation of per-class lwlrap.



def calculate_per_class_lwlrap(truth, scores):

  """Calculate label-weighted label-ranking average precision.

  

  Arguments:

    truth: np.array of (num_samples, num_classes) giving boolean ground-truth

      of presence of that class in that sample.

    scores: np.array of (num_samples, num_classes) giving the classifier-under-

      test's real-valued score for each class for each sample.

  

  Returns:

    per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each 

      class.

    weight_per_class: np.array of (num_classes,) giving the prior of each 

      class within the truth labels.  Then the overall unbalanced lwlrap is 

      simply np.sum(per_class_lwlrap * weight_per_class)

  """

  assert truth.shape == scores.shape

  num_samples, num_classes = scores.shape

  # Space to store a distinct precision value for each class on each sample.

  # Only the classes that are true for each sample will be filled in.

  precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))

  for sample_num in range(num_samples):

    pos_class_indices, precision_at_hits = (

      _one_sample_positive_class_precisions(scores[sample_num, :], 

                                            truth[sample_num, :]))

    precisions_for_samples_by_classes[sample_num, pos_class_indices] = (

        precision_at_hits)

  labels_per_class = np.sum(truth > 0, axis=0)

  weight_per_class = labels_per_class / float(np.sum(labels_per_class))

  # Form average of each column, i.e. all the precisions assigned to labels in

  # a particular class.

  per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) / 

                      np.maximum(1, labels_per_class))

  # overall_lwlrap = simple average of all the actual per-class, per-sample precisions

  #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)

  #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples

  #                = np.sum(per_class_lwlrap * weight_per_class)

  return per_class_lwlrap, weight_per_class
# Calculate the overall lwlrap using sklearn.metrics function.



def calculate_overall_lwlrap_sklearn(truth, scores):

  """Calculate the overall lwlrap using sklearn.metrics.lrap."""

  # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.

  sample_weight = np.sum(truth > 0, axis=1)

  nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)

  overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(

      truth[nonzero_weight_sample_indices, :] > 0, 

      scores[nonzero_weight_sample_indices, :], 

      sample_weight=sample_weight[nonzero_weight_sample_indices])

  return overall_lwlrap
# Accumulator object version.



class lwlrap_accumulator(object):

  """Accumulate batches of test samples into per-class and overall lwlrap."""  



  def __init__(self):

    self.num_classes = 0

    self.total_num_samples = 0

  

  def accumulate_samples(self, batch_truth, batch_scores):

    """Cumulate a new batch of samples into the metric.

    

    Args:

      truth: np.array of (num_samples, num_classes) giving boolean

        ground-truth of presence of that class in that sample for this batch.

      scores: np.array of (num_samples, num_classes) giving the 

        classifier-under-test's real-valued score for each class for each

        sample.

    """

    assert batch_scores.shape == batch_truth.shape

    num_samples, num_classes = batch_truth.shape

    if not self.num_classes:

      self.num_classes = num_classes

      self._per_class_cumulative_precision = np.zeros(self.num_classes)

      self._per_class_cumulative_count = np.zeros(self.num_classes, 

                                                  dtype=np.int)

    assert num_classes == self.num_classes

    for truth, scores in zip(batch_truth, batch_scores):

      pos_class_indices, precision_at_hits = (

        _one_sample_positive_class_precisions(scores, truth))

      self._per_class_cumulative_precision[pos_class_indices] += (

        precision_at_hits)

      self._per_class_cumulative_count[pos_class_indices] += 1

    self.total_num_samples += num_samples



  def per_class_lwlrap(self):

    """Return a vector of the per-class lwlraps for the accumulated samples."""

    return (self._per_class_cumulative_precision / 

            np.maximum(1, self._per_class_cumulative_count))



  def per_class_weight(self):

    """Return a normalized weight vector for the contributions of each class."""

    return (self._per_class_cumulative_count / 

            float(np.sum(self._per_class_cumulative_count)))



  def overall_lwlrap(self):

    """Return the scalar overall lwlrap for cumulated samples."""

    return np.sum(self.per_class_lwlrap() * self.per_class_weight())
#https://www.kaggle.com/voglinio/keras-2d-model-5-fold-log-specgram-curated-only

truth = test_lable_bool

scores = res_test

print("lwlrap from sklearn.metrics =", calculate_overall_lwlrap_sklearn(truth, scores))
truth = train_lable_bool

scores = res_train

print("lwlrap from sklearn.metrics =", calculate_overall_lwlrap_sklearn(truth, scores))
truth = cv_lable_bool

scores = res_cv

print("lwlrap from sklearn.metrics =", calculate_overall_lwlrap_sklearn(truth, scores))
sub_dataframe = pd.DataFrame({'fname':os.listdir('../input/sc2datawithoutaug/sub2/sub2')})



sub_datagen=ImageDataGenerator(rescale=1./255.)

sub_generator=sub_datagen.flow_from_dataframe(

    dataframe=sub_dataframe,

    directory="../input/sc2datawithoutaug/sub2/sub2",

    x_col="fname",

    y_col=None,

    batch_size=64,

    seed=42,

    shuffle=False,

    class_mode=None,

    target_size=(64,64))
STEP_SIZE_SUB=sub_generator.n//sub_generator.batch_size
sub_generator.reset()

res_sub=custom_densenet169_model.predict_generator(sub_generator,

#steps=STEP_SIZE_TEST,

verbose=1)
res_sub.shape
submit_data=pd.DataFrame(res_sub.astype("float64"), columns=list(mlb_train.classes_))
submit_data.insert(0, 'fname', os.listdir('../input/sc2datawithoutaug/sub2/sub2'))

submit_data["fname"]=submit_data["fname"].apply(lambda x: x.split(".")[0]+".wav")

submit_data.head()
submit_data.to_csv("submissionCurated.csv",index=False )