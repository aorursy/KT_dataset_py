import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # basic plotting

import seaborn as sns # additional plotting functionality

import os

from tqdm import tqdm

from glob import glob

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

print(os.listdir("../input"))
# load data

df = pd.read_csv('../input/data/Data_Entry_2017.csv')



# see how many observations there are

num_obs = len(df)

print('Number of observations:',num_obs)



df.head(5) 
my_glob = glob('../input/data/images*/images/*.png')

print('Number of Observations: ', len(my_glob)) # check to make sure I've captured every pathway, should equal 112,120
full_img_paths = {os.path.basename(x): x for x in my_glob}

df['full_path'] = df['Image Index'].map(full_img_paths.get)
num_unique_labels = df['Finding Labels'].nunique()

print('Number of unique labels:',num_unique_labels)



count_per_unique_label = df['Finding Labels'].value_counts() # get frequency counts per label

df_count_per_unique_label = count_per_unique_label.to_frame() # convert series to dataframe for plotting purposes



print(df_count_per_unique_label) # view tabular results

plt.figure(figsize = (12,8))

sns.barplot(x = df_count_per_unique_label.index[:20], y="Finding Labels", data=df_count_per_unique_label[:20], color = "blue"), plt.xticks(rotation = 90) # visualize results graphically
# Selecting only 14 classes excluding 'No finding'

classes = ['Atelectasis',

                'Consolidation',

                'Infiltration', 

                'Pneumothorax', 

                'Edema', 

                'Emphysema',

                'Fibrosis', 

                'Effusion', 

                'Pneumonia',

                'Pleural_Thickening',

                'Cardiomegaly',

                'Nodule', 

                'Mass', 

                'Hernia'] # taken from paper



# One Hot Encoding of Finding Labels to classes

for label in classes:

    df[label] = df['Finding Labels'].map(lambda result: 1.0 if label in result else 0)

df.head(5) # check the data, looking good!
# now, let's see how many cases present for each of of our 14 clean classes (which excl. 'No Finding')

clean_labels = df[classes].sum().sort_values(ascending= False) # get sorted value_count for clean labels



# plot cases using seaborn barchart

clean_labels_df = clean_labels.to_frame() # convert to dataframe for plotting purposes

plt.figure(figsize = (12, 8))

sns.barplot(x = clean_labels_df.index[::], y= 0, data = clean_labels_df[::], color = "blue"), plt.xticks(rotation = 90) # visualize results graphically
data = df[classes]
df['labels'] = data.apply(lambda row: np.argmax(row) if np.sum(row)>0 else -1, axis = 1)
df.head()
from tqdm import tqdm

def cut_labels(df, target = 'labels', max_sample_per_class = 100):

    

    data = pd.DataFrame({})

    

    for label in tqdm(df[target].unique()):

        

        temp = df[df[target]==label].iloc[:max_sample_per_class, :]

        data = pd.concat((data, temp),axis = 0)

            

    return data
short_df = cut_labels(df[df.labels>=0], max_sample_per_class = 1000)

short_df.labels.value_counts()
n_row = 14

n_col = 5



fig, ax = plt.subplots(n_row, n_col, figsize = (n_col*4, n_row*4), constrained_layout = True)



for row in tqdm(range(n_row)):

    

    data = df[df.labels ==row]

    

    for col in range(n_col):

        

        ax[row][col].imshow(plt.imread(data.full_path.iloc[n_col*row + col]), cmap = 'bone')

        ax[row][col].set_xticks([])

        ax[row][col].set_yticks([])

        title = data['Finding Labels'].iloc[n_col*row + col]

        ax[row][col].set_title(f'{title}', fontsize = 10)
# split the data into a training and testing set

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(short_df, test_size = 0.2, stratify = short_df.labels, random_state = 1993)



# quick check to see that the training and test set were split properly

print('training set - # of observations: ', len(train_df))

print('test set - # of observations): ', len(test_df))

print('prior, full data set - # of observations): ', len(short_df))
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.1,

        zoom_range=0.1,

        rotation_range=20,

        width_shift_range=0.1,

        height_shift_range=0.1,

        horizontal_flip=True)



test_gen = ImageDataGenerator(

        rescale=1./255)
train_df['labels'] = train_df['labels'].astype(str)

test_df['labels'] = test_df['labels'].astype(str)
image_size = (128, 128)

train_generator  = train_gen.flow_from_dataframe(train_df,

                                                x_col = 'full_path',

                                                y_col = 'labels',

                                                batch_size = 64,

                                                target_size = image_size)



test_generator  = test_gen.flow_from_dataframe(test_df,

                                                x_col = 'full_path',

                                                y_col = 'labels',

                                                batch_size = 64,

                                                target_size = image_size)


import itertools

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import pandas.util.testing as tm

from sklearn import metrics

import seaborn as sns

sns.set()



plt.rcParams["font.family"] = 'DejaVu Sans'



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues,

                          save = False):

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.grid(b=False)

    if save == True:

      plt.savefig('Confusion Matrix.png', dpi = 900)



def plot_roc_curve(y_true, y_pred, classes):



    from sklearn.metrics import roc_curve, auc



    # create plot

    fig, c_ax = plt.subplots(1,1, figsize = (9, 9))

    for (i, label) in enumerate(classes):

        fpr, tpr, thresholds = roc_curve(y_true[:,i].astype(int), y_pred[:,i])

        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (label, auc(fpr, tpr)))



    # Set labels for plot

    c_ax.legend()

    c_ax.set_xlabel('False Positive Rate')

    c_ax.set_ylabel('True Positive Rate')

    c_ax.set_title('Roc AUC Curve')

    

    

# test model performance

from datetime import datetime

import matplotlib.pyplot as plt





def test_model(model, test_generator, y_test, class_labels, cm_normalize=True, \

                 print_cm=True):

    

    results = dict()



    print('\nPredicting test data')

    test_start_time = datetime.now()

    y_pred_original = model.predict_generator(test_generator,verbose=1)

    # y_pred = (y_pred_original>0.5).astype('int')



    y_pred = np.argmax(y_pred_original, axis = 1)

    # y_test = np.argmax(testy, axis= 1)

    #y_test = np.argmax(testy, axis=-1)

    

    test_end_time = datetime.now()

    print('Done \n \n')

    results['testing_time'] = test_end_time - test_start_time

    print('testing time(HH:MM:SS:ms) - {}\n\n'.format(results['testing_time']))

    results['predicted'] = y_pred

    y_test = y_test.astype(int) # sparse form not categorical

    



    # balanced_accuracy

    from sklearn.metrics import balanced_accuracy_score

    balanced_accuracy = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)

    print('---------------------')

    print('| Balanced Accuracy  |')

    print('---------------------')

    print('\n    {}\n\n'.format(balanced_accuracy))



    

    # calculate overall accuracty of the model

    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)

    # store accuracy in results

    results['accuracy'] = accuracy

    print('---------------------')

    print('|      Accuracy      |')

    print('---------------------')

    print('\n    {}\n\n'.format(accuracy))

    



    # get classification report

    print('-------------------------')

    print('| Classifiction Report |')

    print('-------------------------')

    classification_report = metrics.classification_report(y_test, y_pred)

    # store report in results

    results['classification_report'] = classification_report

    print(classification_report)

    

    #roc plot

    plot_roc_curve(tf.keras.utils.to_categorical(y_test), y_pred_original, class_labels)

    

   



    # confusion matrix

    cm = metrics.confusion_matrix(y_test, y_pred)

    results['confusion_matrix'] = cm

    if print_cm: 

        print('--------------------')

        print('| Confusion Matrix |')

        print('--------------------')

        print('\n {}'.format(cm))

        

    # plot confusin matrix

    plt.figure(figsize=(16,12))

    plt.grid(b=False)

    plot_confusion_matrix(cm, classes=class_labels, normalize=True, title='Normalized confusion matrix')

    plt.show()



    

    # add the trained  model to the results

    results['model'] = model

    

    return





from tensorflow.keras.callbacks import Callback

class MyLogger(Callback):

  

  def __init__(self, test_generator, y_test, class_labels):

    super(MyLogger, self).__init__()

    self.test_generator = test_generator

    self.y_test = y_test

    self.class_labels = class_labels

    

  def on_epoch_end(self, epoch, logs=None):

    test_model(self.model, self.test_generator, self.y_test, self.class_labels)



#   def _implements_train_batch_hooks(self): return True

#   def _implements_test_batch_hooks(self): return True

#   def _implements_predict_batch_hooks(self): return True
from tensorflow.keras.callbacks import *

def get_callbacks():

    

    filepath = 'best_model_multiclass_128.h5'

    callback1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    callback2 = MyLogger(test_generator,

                    test_df[classes].values.argmax(axis = 1), 

                    classes)

    



    return [callback1 ,callback2]
from tensorflow.keras.layers import *

from tensorflow.keras.models import *
def Residual_Unit(input_tensor, nb_of_input_channels, max_dilation, number_of_units):

    

  for i in range(number_of_units):

    x1 = Conv2D(nb_of_input_channels*2, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(input_tensor)

    x1 = BatchNormalization()(x1)

  

    a = []



    for i in range(1, max_dilation+1):

      temp = DepthwiseConv2D( kernel_size=(3,3), dilation_rate = (i,i), padding = 'same', activation= 'relu')(x1)

      temp = BatchNormalization()(temp)

      a.append(temp)



    x = Concatenate(axis= -1)(a)

    x = Conv2D(nb_of_input_channels, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(x)

    x = BatchNormalization()(x)



    x = Add()([x, input_tensor])



    input_tensor = x

  

  return x
def Shifter_Unit(input_tensor, nb_of_input_channels, max_dilation):

    x1 = Conv2D(nb_of_input_channels*4, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(input_tensor)

    x1 = BatchNormalization()(x1)



    a = []



    for i in range(1, max_dilation+1):

      temp = DepthwiseConv2D( kernel_size=(3,3), dilation_rate = (i,i), padding = 'same', activation= 'relu')(x1)

      temp = MaxPool2D(pool_size=(2,2), padding = 'same')(temp)

      temp = BatchNormalization()(temp)

      a.append(temp)



    x = Concatenate(axis= -1)(a)



    x = Conv2D(nb_of_input_channels*2, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(x)

    x = BatchNormalization()(x)



    return x
from tensorflow.keras.optimizers import Adam



#Network:

  

def Network128(input_shape, nb_class, depth):

  xin = Input(shape= input_shape)



  x = Conv2D(16, kernel_size = (5,5), strides= (1,1), padding = 'same', activation='relu')(xin)

  x = BatchNormalization()(x)



  x = Conv2D(32, kernel_size = (3,3), strides= (2,2), padding = 'same', activation='relu')(x)

  x = BatchNormalization()(x)

  

##Max Dilation rate will be vary in the range (1,5). 



# Max Dilation rate is 5 for tensor (64x64x32)

  x = Residual_Unit(input_tensor=x, nb_of_input_channels=32, max_dilation=5, number_of_units=depth)

  x = Shifter_Unit(input_tensor=x, nb_of_input_channels=32, max_dilation=5)





# Max Dilation rate is 4 for (32x32x64)

  x = Residual_Unit(input_tensor=x, nb_of_input_channels=64, max_dilation=4, number_of_units=depth)

  x = Shifter_Unit(input_tensor=x, nb_of_input_channels=64, max_dilation=4)



# Max Dilation rate is 3 for (16x16x128)

  x = Residual_Unit(input_tensor=x, nb_of_input_channels=128, max_dilation=3, number_of_units=depth)

  x = Shifter_Unit(input_tensor=x, nb_of_input_channels=128, max_dilation=3)



# Max Dilation rate is 2 for (8x8x256)

  x = Residual_Unit(input_tensor=x, nb_of_input_channels=256, max_dilation=2, number_of_units=depth)



  x = GlobalAveragePooling2D()(x)



  x = Dense(128, activation='relu')(x)

  x = Dense(64, activation='relu')(x)



  x = Dense(nb_class, activation= 'softmax')(x)



  model = Model(xin, x)



  model.compile(loss='categorical_crossentropy', optimizer = Adam(lr = 1e-3), metrics = ['accuracy'])



  return model

from sklearn.utils import class_weight

 

 

class_weights = class_weight.compute_class_weight('balanced',

                                                 np.unique(train_df.labels.astype(int)),

                                                 train_df.labels.astype(int))

class_weight =  dict(zip(np.sort(train_df.labels.astype(int).unique()), class_weights))

class_weight
model = Network128(input_shape = (128, 128, 3), nb_class = 14, depth = 5)

model.summary()
model.fit_generator(generator = train_generator,

                    steps_per_epoch = train_generator.samples/train_generator.batch_size,

                    epochs = 45,

                    validation_data = test_generator,

                    validation_steps = test_generator.samples/test_generator.batch_size,

                    class_weight = class_weight,

                    callbacks = get_callbacks()

                   )
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

acc = model.history.history['accuracy']

val_acc = model.history.history['val_accuracy']

loss = model.history.history['loss']

val_loss = model.history.history['val_loss']



epochs = range(0,len(acc))

fig = plt.gcf()

fig.set_size_inches(16, 8)



plt.plot(epochs, acc, 'r', label='Training accuracy',marker = "o")

plt.plot(epochs, val_acc, 'b', label='Validation accuracy',marker = "o")

plt.title('Training and validation accuracy')

plt.xticks(np.arange(0, len(acc), 10))

plt.legend(loc=0)

plt.figure()



fig = plt.gcf()

fig.set_size_inches(16, 8)

plt.plot(epochs, loss, 'r', label='Training Loss',marker = "o")

plt.plot(epochs, val_loss, 'b', label='Validation Loss',marker = "o")

plt.title('Training and validation Loss')

plt.xticks(np.arange(0, len(acc), 10))

plt.legend(loc=0)

#plt.savefig('Multiclass Model .png')

plt.figure()

plt.show()

from tensorflow.keras.models import load_model



best_model = load_model('/kaggle/working/best_model_multiclass_128.h5')
test_model(best_model, test_generator, y_test = test_df[classes].values.argmax(axis = 1), class_labels = classes)