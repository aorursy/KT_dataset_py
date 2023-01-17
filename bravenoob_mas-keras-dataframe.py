# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier

from catboost import CatBoostClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder,OrdinalEncoder, StandardScaler,KBinsDiscretizer

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD



# Evaluation

from sklearn.metrics import cohen_kappa_score,make_scorer

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



#ignore warnings

import warnings

warnings.filterwarnings('ignore')

from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore", category=ConvergenceWarning)



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')
import time 



#let's also import the abstract base class for our callback

from tensorflow.keras.callbacks import Callback



#defining the callback

class TimerCallback(Callback):

    

    def __init__(self, maxExecutionTime):

        

# Arguments:

#     maxExecutionTime (number): Time in minutes. The model will keep training 

#                                until shortly before this limit

#                                (If you need safety, provide a time with a certain tolerance)

        

        self.maxExecutionTime = maxExecutionTime * 60

    

    

    #Keras will call this when training begins

    def on_train_begin(self, logs):

        self.startTime = time.time()

        self.longestTime = 0            #time taken by the longest epoch or batch

        self.lastTime = self.startTime  #time when the last trained epoch or batch was finished



    #this is our custom handler that will be used in place of the keras methods:

        #`on_batch_end(batch,logs)` or `on_epoch_end(epoch,logs)`

    def on_epoch_end(self, epoch, logs):

        

        currentTime      = time.time()                           

        self.elapsedTime = currentTime - self.startTime    #total time taken until now

        thisTime         = currentTime - self.lastTime     #time taken for the current epoch

                                                               #or batch to finish

        

        self.lastTime = currentTime

        

        #verifications will be made based on the longest epoch or batch

        if thisTime > self.longestTime:

            self.longestTime = thisTime

        

        

        #if the (assumed) time taken by the next epoch or batch is greater than the

            #remaining time, stop training

        remainingTime = self.maxExecutionTime - self.elapsedTime

        if remainingTime < self.longestTime:

            

            self.model.stop_training = True  #this tells Keras to not continue training

            print("\n\nTimerCallback: Finishing model training before it takes too much time. (Elapsed time: " + str(self.elapsedTime/60.) + " minutes )\n\n")
train_df = pd.read_csv("../input/petfinder-images-train-test-valid-realtest/images_train_test_valid_realtest/images_train_test_valid_realtest/train_df.csv")

valid_df = pd.read_csv("../input/petfinder-images-train-test-valid-realtest/images_train_test_valid_realtest/images_train_test_valid_realtest/val_df.csv")

train_data_dir = '../input/petfinder-images-train-test-valid-realtest/images_train_test_valid_realtest/images_train_test_valid_realtest/train'

valid_data_dir = '../input/petfinder-images-train-test-valid-realtest/images_train_test_valid_realtest/images_train_test_valid_realtest/valid'



test_df = pd.read_csv("../input/petfinder-images-train-test-valid-realtest/images_train_test_valid_realtest/images_train_test_valid_realtest/test_df.csv")

test_data_dir = '../input/petfinder-images-train-test-valid-realtest/images_train_test_valid_realtest/images_train_test_valid_realtest/test'
import tensorflow as tf

print(tf.__version__)
from tensorflow.keras.applications import NASNetLarge

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout

from tensorflow.keras.optimizers import Adam, Nadam



num_classes = 5

model = NASNetLarge(weights='imagenet', include_top=False, pooling='avg')



my_new_model = Sequential()

my_new_model.add(model)

my_new_model.add(Dense(512, activation="relu"))

my_new_model.add(Dropout(rate=0.25))

my_new_model.add(Dense(256, activation="relu"))

my_new_model.add(Dropout(rate=0.25))                 

my_new_model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = False



# optimizer

# descent optimizer (adam lr defaut = 0.001)

my_new_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])



print(my_new_model.summary())
from tensorflow.keras.applications.nasnet import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.callbacks import Callback



image_size = 331

batch_size = 64

nb_epochs = 100



# steps_per_epoch: number of yields (batches) before a epoch is over

# ceil(num_samples / batch_size)

# epochs: Number of epochs to train the model. An epoch is an iteration over the entire data provided

# class_weight: Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). 

#  This can be useful to tell the model to "pay more attention" to samples from an under-represented class.



shift = 0.2



#brightness_range=[0.8,1.2],

#zoom_range=[0.8,1.2],

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,zoom_range=[0.9,1.1],

                                   width_shift_range=shift, height_shift_range=shift, 

                                   horizontal_flip=True)



#liefert die trainingsdaten als iterator

# image aug funktioniert so, dass für den aktuellen batch die bilder geändert werden. Nicht dass es mehr Bilder gibt.

# Somit wird für jede Epoche mit anderen Bildenr trainiert.

train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(image_size, image_size),

    batch_size=batch_size,

    class_mode='categorical')



# keine image augmenation für validierung

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)



#liefert die Validationdaten als iterator

validation_generator = test_datagen.flow_from_directory(

    valid_data_dir,

    target_size=(image_size, image_size),

    batch_size=batch_size,

    class_mode='categorical')



STEP_SIZE_TRAIN=math.ceil(train_generator.n//train_generator.batch_size)

STEP_SIZE_VALID=math.ceil(validation_generator.n//validation_generator.batch_size)



print(STEP_SIZE_TRAIN)

print(STEP_SIZE_VALID)





# Configure the TensorBoard callback and fit the model

tensorboard_callback = TensorBoard("logs")



# Early stopping against overfit

earlystopping_callback = EarlyStopping(patience=batch_size/10, monitor='val_acc', mode='auto', restore_best_weights=True)

        

# save best model

mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='auto', save_best_only=True)



class_weights = {0: 8.80595533,

                1: 1.01597481,

                2: 0.72513282,

                3: 0.74640867,

                4: 0.84505298}





#import multiprocessing

#multiprocessing.cpu_count()



timerCallback = TimerCallback(500)



history = my_new_model.fit_generator(

    train_generator,

    steps_per_epoch = STEP_SIZE_TRAIN,

    validation_data = validation_generator, 

    validation_steps = STEP_SIZE_VALID,

    epochs = nb_epochs,

    class_weight = class_weights,

    callbacks=[mc,timerCallback],

    workers = 2,

    use_multiprocessing = False,

    max_queue_size = 40)
import matplotlib.pyplot as plt

def plot_training(history):

    # Plot training & validation accuracy values

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('Model accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()



    # Plot training & validation loss values

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()



    plt.savefig('acc_vs_epochs.png')

    

plot_training(history)
from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import load_model

from tensorflow.keras.applications.nasnet import preprocess_input, decode_predictions



saved_model = load_model('../working/best_model.h5')
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.callbacks import Callback



image_size = 331

batch_size = 64



train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)



#liefert die traindaten als iterator

test_generator = train_datagen.flow_from_directory(

    test_data_dir,

    target_size=(image_size, image_size),

    batch_size=batch_size,

    class_mode=None,

    shuffle=False)



test_generator.reset()



test_preds = saved_model.predict_generator(test_generator)

test_results=pd.DataFrame({"Filename":test_generator.filenames,

                      "Predictions":test_preds.argmax(axis=-1)})

test_results.Predictions.value_counts()
def get_all_speed_from_predictions(df, PetID):

    values = df[df['Filename'].str.contains(PetID)]['Predictions'].values

    preds = values.tolist()

    mean = values.mean()

    if math.isnan(mean):

        mean = 4 # if no photo is available, default is Speed 4

    else: 

        mean = int(round(mean, 0))

    return pd.Series([preds.count(0),preds.count(1), preds.count(2),preds.count(3),preds.count(4), mean], 

                     index =['AdaptionSpeed0', 'AdaptionSpeed1', 'AdaptionSpeed2', 'AdaptionSpeed3', 'AdaptionSpeed4', 'mean']) 
test_result_speed=pd.DataFrame()

test_result_speed['PetID']=test_df['PetID']

test_result_speed['AdoptionSpeed']=test_df['AdoptionSpeed']

test_result_speed = test_result_speed.merge(test_result_speed.PetID.apply(lambda x: get_all_speed_from_predictions(test_results, x)), 

    left_index=True, right_index=True)
test_result_speed.AdoptionSpeed.value_counts()
#Machine Learning Algorithm (MLA) Selection and Initialization

MLA = [

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),

    

    #GLM

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    #Navies Bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    neighbors.KNeighborsClassifier(),

    

    #SVM

    svm.LinearSVC(),

    

    #Trees    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),



    #xgboost: 

    XGBClassifier(),

    

    CatBoostClassifier(verbose=0)

    ]
def train_model(data, MLA_list = MLA):

    

    target = data['AdoptionSpeed']

    X_train = data.drop(['AdoptionSpeed'],axis=1)

    

    MLA_columns = ['MLA Name', 'MLA Parameters','MLA cohen_kappa_score','MLA Time']

    MLA_compare = pd.DataFrame(columns = MLA_columns)



    MLA_predict = data['AdoptionSpeed']

    

    row_index = 0

    for alg in MLA_list:



        MLA_name = alg.__class__.__name__

        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    

        kf = StratifiedKFold(n_splits=10, shuffle=True)

        kappa_score = make_scorer(cohen_kappa_score, weights='quadratic')

        cv_results = model_selection.cross_validate(alg, X_train, target, cv  = kf, scoring=kappa_score )

        

        MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()

        MLA_compare.loc[row_index, 'MLA cohen_kappa_score'] = cv_results['test_score'].mean() 

             

        #MLA_predict[MLA_name] = alg.predict(X_train)

        row_index+=1



    MLA_compare.sort_values(by = ['MLA cohen_kappa_score'], ascending = False, inplace = True)

    sns.barplot(x='MLA cohen_kappa_score', y = 'MLA Name', data = MLA_compare, color = 'b')

    plt.title('Machine Learning Algorithm Accuracy Score \n')

    plt.xlabel('Accuracy Score (%)')

    plt.ylabel('Algorithm')

    

    return MLA_compare
predictAndTargetColumns = ['AdoptionSpeed', 'AdaptionSpeed0', 'AdaptionSpeed1', 'AdaptionSpeed2', 'AdaptionSpeed3', 'AdaptionSpeed4', 'mean']



classifier_performance = train_model(test_result_speed[predictAndTargetColumns])
classifier_performance
target = test_result_speed['AdoptionSpeed']

X_test = test_result_speed['mean']



cohen_kappa_score(X_test, target, weights='quadratic')
useColumns = ['AdaptionSpeed0', 'AdaptionSpeed1', 'AdaptionSpeed2', 'AdaptionSpeed3', 'AdaptionSpeed4', 'mean']

X_test = test_result_speed[useColumns]



#classifier = CatBoostClassifier(verbose=0)

# train the model

#classifier.fit(X_test, target)
{'colsample_bytree': 0.3,

 'gamma': 0.4,

 'learning_rate': 0.3,

 'max_depth': 5,

 'min_child_weight': 3}





classifier =XGBClassifier().fit(X_test, target)
realtest_df = pd.read_csv('../input/petfinder-images-train-test-valid-realtest/images_train_test_valid_realtest/images_train_test_valid_realtest/realtest/test.csv')

realtest_data_dir = '../input/petfinder-images-train-test-valid-realtest/images_train_test_valid_realtest/images_train_test_valid_realtest/realtest'
realtest_df.shape
realtest_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)



#liefert die testdaten als iterator

realtest_generator = realtest_datagen.flow_from_directory(

    realtest_data_dir, 

    target_size=(image_size, image_size),

    batch_size=batch_size,

    class_mode=None,

    shuffle=False)



realtest_generator.reset()



realtest_preds = saved_model.predict_generator(realtest_generator)

realtest_results=pd.DataFrame({"Filename":realtest_generator.filenames,

                      "Predictions":realtest_preds.argmax(axis=-1)})
realtest_result_speed=pd.DataFrame()

realtest_result_speed['PetID']=realtest_df['PetID']

realtest_result_speed = realtest_result_speed.merge(realtest_result_speed.PetID.apply(lambda x: get_all_speed_from_predictions(realtest_results, x)), 

    left_index=True, right_index=True)
useColumns = ['AdaptionSpeed0', 'AdaptionSpeed1', 'AdaptionSpeed2', 'AdaptionSpeed3', 'AdaptionSpeed4', 'mean']



submit=pd.DataFrame()

submit['PetID']=realtest_result_speed['PetID']

submit['AdoptionSpeed']=classifier.predict(realtest_result_speed[useColumns])

submit['AdoptionSpeed']=submit['AdoptionSpeed'].astype(int)

submit.to_csv('submission.csv',index=False)