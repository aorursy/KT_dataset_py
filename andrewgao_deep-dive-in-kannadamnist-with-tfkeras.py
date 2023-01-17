import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
#This csv file saved all the validation result of my previous training.



old_result_df = pd.read_csv('../input/kmnist-data/Old_training_result.csv')

old_result_df.head(5)
def display_single_fold_val_accuracy(final_df, fold):

    epochs = [6,18,30]

    color = ['y','g','b']

    plt.figure(figsize=(12,6))

    for c, e in zip(color,epochs):

        for fold in [fold]:

            df = final_df[(final_df['fold'] == fold) & (final_df['epoch'] == e)]

            label = 'fold:'+str(fold)+', epoch:' +str(e) 

            plt.plot(df['class'].values, df['valid_accuracy'].values, label = label, c =c)

    plt.plot(np.arange(0,10,1), np.ones(10), '^', label = 'Top accuracy', color='c')

    plt.plot(np.arange(0,10,1), np.ones(10)*0.995, '--', label = 'baseline', color='r')

    plt.xticks(np.arange(0,10,1))

    plt.title('Valid accuracy on fold'+str(fold)+' model')

    plt.legend()

    plt.show()



def display_top_accuracy_on_final_epoch(final_df):

    color = ['y','g','b']

    plt.figure(figsize=(12,6))

    for c,f in zip(color, [1,2,3]):

        df = final_df[(final_df['fold'] == f) & (final_df['epoch'] == 30)]

        label = 'fold:'+str(f)+', epoch:' +str(30) 

        plt.plot(df['class'].values, df['valid_accuracy'].values, label = label, c =c)

    plt.plot(np.arange(0,10,1), np.ones(10), '^', label = 'Top accuracy', color='c')

    plt.plot(np.arange(0,10,1), np.ones(10)*0.995, '--', label = 'baseline', color='r')

    plt.xticks(np.arange(0,10,1))

    plt.title('Valid accuracy on all models at epoch 30')

    plt.legend()

    plt.show()
display_single_fold_val_accuracy(old_result_df, 1)
display_single_fold_val_accuracy(old_result_df, 2)
display_single_fold_val_accuracy(old_result_df, 3)
#This csv file saved all the validation result of my current training.

new_result_df = pd.read_csv('../input/kmnist-data/New_training_result.csv')

new_result_df.head(5)
plt.figure(figsize=(12,8))

for fold in range(1,9,1):

    df = new_result_df[new_result_df['fold'] == fold]

    plt.plot(df.Class.values-1, df.Valid_accuracy.values, label = 'fold_'+str(fold)+'_model_epoch_'+str(30))

plt.plot(np.arange(0,10,1), np.ones(10)*0.995, '--', label = 'baseline', c='r')

plt.plot(np.arange(0,10,1), np.ones(10), '^', label = 'Top accuracy', color='c')

plt.legend()

plt.xticks(np.arange(0,10,1))

plt.show()
import tensorflow as tf

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import seaborn as sn

import albumentations as albu

from sklearn.model_selection import train_test_split, KFold

from tqdm import tqdm_notebook

import gc

import os

import warnings 

warnings.filterwarnings('ignore')

main_dir = '../input/Kannada-MNIST/'

tf.keras.__version__
pretrain_weights_path = [

    '../input/kmnist-data/0_model.hdf5',

    '../input/kmnist-data/1_model.hdf5',

    '../input/kmnist-data/2_model.hdf5',

    '../input/kmnist-data/3_model.hdf5',

    '../input/kmnist-data/4_model.hdf5',

    '../input/kmnist-data/5_model.hdf5',

    '../input/kmnist-data/6_model.hdf5',

    '../input/kmnist-data/7_model.hdf5',

]





uptrain = False

submit = True

num_classes = 10

num_features = (28,28,1)

batch_size = 1024

lr = 3e-4

epochs = 33





k_fold_split = 8
train_df = pd.read_csv(main_dir + 'train.csv')

valid_df = pd.read_csv(main_dir + 'Dig-MNIST.csv')

test_df = pd.read_csv(main_dir + 'test.csv')



#Check out some training data

train_df.head(5)
#Extract the label from training dataframe and discard the label column

train_label = train_df['label']

test_indices = test_df['id']



train_df = train_df.drop(['label'], axis = 1)

test_df = test_df.drop(['id'], axis = 1)



#Convert dataframe into numpy array 

train_x = train_df.values

train_y = train_label.values



test_x = test_df.values



print("shape of train_x :", train_x.shape)

print("shape of train_y :", train_y.shape)

print("shape of test_x :", test_x.shape)
train_x = train_x.reshape(-1,28,28,1)

# One-hot encode the original label

train_y = tf.keras.utils.to_categorical(train_y, num_classes)

test_x = test_x.reshape(-1,28,28,1)
#check some image data

temp_imgs = train_x[8:12]

temp_labels = train_y[8:12]



nrows = 2

ncols = 2

plt.figure(figsize=(6,6))



for idx, (img, label) in enumerate(zip(temp_imgs, temp_labels)):

    plt.subplot(nrows, ncols, idx+1)

    plt.imshow(np.squeeze(img,axis=2), cmap = 'gray')

    plt.title("label : " + str(np.argmax(label)))

    plt.axis('off')

plt.show()
#check the quantity of each labels

label_counts = train_label.value_counts().reset_index()

#rename the columns of label_counts

label_counts.columns = ['label', 'quantity']

#sort the value in label_counts on label column

label_counts = label_counts.sort_values('label')



plt.figure(figsize = (8,4))

plt.bar(label_counts['label'], label_counts['quantity'])

plt.xlabel('label')

plt.ylabel('quantity')

plt.title('label quantity in training dataset')

plt.show()

#x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size = 0.2, random_state = 2019)
def display_aug_effect(img, aug, repeat=3, aug_item = 'rotate'):

    '''

    img : input image for display

    aug : augmentation object to perform image multiplication

    repeat : how much time you want to perfrom multiplication

    aug_item : certain multiplication you want to apply to input image

    '''

    plt.figure(figsize=(int(4*(repeat+1)),4))

    plt.subplot(1,repeat+1, 1)

    plt.imshow(img, cmap='gray')

    plt.title('original image')

    

    for i in range(repeat):

        plt.subplot(1, repeat+1, i+2)

        temp_aug_img = aug(image = img.astype('uint8'))['image']

        plt.imshow(temp_aug_img, cmap='gray')

        plt.title(aug_item + ' : ' + str(i+1))

    

    plt.axis('off')

    plt.show()
temp_aug = albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=20, shift_limit=0.15, p=1, border_mode=0)

display_aug_effect(np.squeeze(temp_imgs[0], axis=2), temp_aug, aug_item = 'ShiftScaleRotate')
temp_aug = albu.GridDistortion(p=1)

display_aug_effect(np.squeeze(temp_imgs[1], axis=2), temp_aug, aug_item = 'GridDistortion')
temp_aug = albu.OneOf([ albu.RandomBrightness(limit=10), albu.RandomGamma(gamma_limit=(80, 120)), albu.RandomContrast(limit=1.5) ], p=1)

display_aug_effect(np.squeeze(temp_imgs[2], axis=2), temp_aug, aug_item = 'Gamma/Brightness/Contrast')
temp_aug = albu.RandomCrop(height=24,width=24,p=1)

display_aug_effect(np.squeeze(temp_imgs[3], axis=2), temp_aug, aug_item = 'RandomCrop')
class InputGenerator(tf.keras.utils.Sequence):

    

    def __init__(self,

                 x,

                 y=None,

                 aug=None,

                 batch_size=128,

                 training=True):

        

        self.x = x

        self.y = y

        self.aug = aug

        self.batch_size = batch_size

        self.training = training

        self.indices = range(len(x))

    

    def __len__(self):

        return len(self.x) // self.batch_size

    

    def __getitem__(self,index):

        

        batch_indices = self.indices[index * self.batch_size : (index+1)*self.batch_size]

        batch_data = self.__get_batch_x(batch_indices)

        

        if self.training:

            batch_label = self.__get_batch_y(batch_indices)

            return batch_data, batch_label

        else:

            return batch_data

    

    def on_epoch_start(self):

        

        if self.training:

            np.random.shuffle(self.indices)

            

    def __get_batch_x(self, batch_indices):

        

        batch_data = []

        for idx in batch_indices:

            cur_data = self.x[idx].astype('uint8')

            cur_data = self.aug(image = cur_data)['image']

            batch_data.append(cur_data)

            

        return np.stack(batch_data)/255.0

    

    def __get_batch_y(self, batch_indices):

        

        batch_label = []

        for idx in batch_indices:

            batch_label.append(self.y[idx])

            

        return np.stack(batch_label)





        

train_aug = albu.Compose([

                    albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=15.0, shift_limit=0.15, p=0.5, border_mode=0, value = 0)]

                    )



valid_aug = albu.Compose([])
def _swish(x):

    '''

    x : input tensor

    

    return : swish activated tensor

    '''

    return tf.keras.backend.sigmoid(x) * x



#helper function of Squeeze and Excitation block.

def _seblock(input_channels=32, se_ratio=2):

    '''

    input_channels : the channels of input tensor

    se_ratio : the ratio for reducing the first fully-conntected layer

    

    return : helper function for entire se block

    '''

    def f(input_x):

        

        reduced_channels = input_channels // se_ratio

        

        x = tf.keras.layers.GlobalAveragePooling2D()(input_x)

        x = tf.keras.layers.Dense(units=reduced_channels, kernel_initializer='he_normal')(x)

        x = tf.keras.layers.Activation(_swish)(x)

        x = tf.keras.layers.Dense(units=input_channels, kernel_initializer='he_normal', activation='sigmoid')(x)

        

        x = tf.keras.layers.multiply([x,input_x])

        

        return x

    return f



def _cn_bn_act(filters=64, kernel_size=(3,3), strides=(1,1)):

    '''

    filters : filter number of convolution layer

    kernel_size : filter/kernel size of convolution layer

    strides : stride size of convolution layer

    

    return : helper function for convolution -> batch normalization -> activation

    '''

    def f(input_x):

        

        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(input_x)

        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation(_swish)(x)

        

        return x

    

    return f





def _dn_bn_act(units=128):

    '''

    units : units for fully-connected layer

    

    return : helper function for fully-connected -> batch normalization -> activation

    '''

    def f(input_x):

        

        x = tf.keras.layers.Dense(units=units, kernel_initializer='he_normal')(input_x)

        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation(_swish)(x)

        return x

    

    return f



def build_model(input_shape = (28,28,1), classes = 10):

    '''

    input_shape : input dimension of single data

    classes : class number of label

    

    return : cnn model

    '''

    input_layer = tf.keras.layers.Input(shape = input_shape)

    

    x = _cn_bn_act(filters=64)(input_layer)

    #x = _seblock(input_channels=64)(x)

    x = _cn_bn_act(filters=64)(x)

    x = _cn_bn_act(filters=64)(x)

        

    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    x = tf.keras.layers.Dropout(0.2)(x)

    

    x = _cn_bn_act(filters=128)(x)

    #x = _seblock(input_channels=128)(x)

    x = _cn_bn_act(filters=128)(x)

    x = _cn_bn_act(filters=128)(x)

    

    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    x = tf.keras.layers.Dropout(0.2)(x)

    

    x = _cn_bn_act(filters=256)(x)    

    x = _seblock(input_channels=256)(x)

    x = _cn_bn_act(filters=256)(x)    

    

    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Flatten()(x)

    

    x = _dn_bn_act(units=256)(x)

    x = _dn_bn_act(units=128)(x)

    output_layer = tf.keras.layers.Dense(units=classes, kernel_initializer='he_normal', activation = 'softmax')(x)

    

    model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])

    return model
def Precision(y_true, y_pred, epsilon=1e-7):

    

    y_true_f = tf.keras.backend.flatten(y_true)

    y_pred_f = tf.keras.backend.flatten(y_pred)

    

    y_pred_f = tf.keras.backend.round(y_pred_f)

    

    TP = tf.keras.backend.sum(y_true_f * y_pred_f)

    FP = tf.keras.backend.sum((1-y_true_f) * y_pred_f)

    

    return TP/(TP+FP+epsilon)



def Recall(y_true, y_pred, epsilon=1e-7):

    

    y_true_f = tf.keras.backend.flatten(y_true)

    y_pred_f = tf.keras.backend.flatten(y_pred)

    

    y_pred_f = tf.keras.backend.round(y_pred_f)

    

    TP = tf.keras.backend.sum(y_true_f * y_pred_f)

    TN = tf.keras.backend.sum(y_true_f * (1-y_pred_f))

    

    return TP/(TP+TN+epsilon)
def symmetric_cross_entropy(alpha=1.0, beta=1.0, epsilon=1e-7):

    def loss(y_true, y_pred):

        

        y_pred_ce = tf.clip_by_value(y_pred, epsilon, 1.0)

        y_true_rce = tf.clip_by_value(y_true, epsilon, 1.0)



        ce = alpha*tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred_ce), axis = -1))

        rce = beta*tf.reduce_mean(-tf.reduce_sum(y_pred * tf.math.log(y_true_rce), axis = -1))

        

        return  ce + rce

    return loss
# Before training, I would like to shuffle the training dataset to make the data more random

# But we also need to shuffle the training label and keep it has same corresponding relation with training dataset.

# So I use permutation in numpy to create the new indices for both new training data & label

permutation = np.random.RandomState(2019).permutation(len(train_x))

train_x = train_x[permutation]

train_y = train_y[permutation]
kf = KFold(n_splits=k_fold_split, random_state=2019)



#list to save all the models we are going to train

model_members = []

check_points_path = []



for idx in range(k_fold_split):

    check_points_path.append(str(idx) + '_model.hdf5')

    

for model_index, (train_indices, valid_indices) in enumerate(kf.split(train_x)):

    

    #data generator for training

    train_aug_gen = InputGenerator(train_x[train_indices], train_y[train_indices], train_aug, batch_size)

    #data generator for validating

    valid_aug_gen = InputGenerator(train_x[valid_indices], train_y[valid_indices], valid_aug, batch_size)

    

    model = build_model()

    

    optimizer = tf.keras.optimizers.RMSprop(lr=lr)

    model.compile(optimizer=optimizer, metrics=['accuracy',Precision,Recall], loss=symmetric_cross_entropy())



    steps_per_epoch = len(train_x[train_indices]) // batch_size

    validation_steps = len(train_x[valid_indices]) // batch_size

    

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-8, mode='min', verbose=2)

    check_point = tf.keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath = check_points_path[model_index], mode='min', save_best_only=True, verbose=2)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    

    training_callbacks = [ check_point, reduce_lr ]

    

    if pretrain_weights_path == None or uptrain == True:

        

        if pretrain_weights_path != None:

            print('*'*10,'Load ',model_index,'-fold pretrain weights','*'*10)

            model.load_weights(pretrain_weights_path[model_index])

            

        

        

        print('*'*30,'Training model ', model_index+1,'*'*30)

        print('Train on ', len(train_indices),' data')

        print('Valid on ', len(valid_indices),' data')



        history = model.fit_generator(generator=train_aug_gen,

                                      steps_per_epoch=steps_per_epoch,

                                      epochs=epochs,

                                      validation_data=valid_aug_gen,

                                      validation_steps=validation_steps,

                                      workers=-1,

                                      verbose=2,

                                      callbacks=training_callbacks)

        

        print('*'*30,'Validating model ', model_index+1,'*'*30)

        val_gen = InputGenerator(train_x[valid_indices], None, albu.Compose([]), 1, training=False)

        

        print('\n\n')

        preds = model.predict_generator(val_gen, workers=-1, verbose=1)

        truths = train_y[valid_indices]

        preds = np.round( np.array(preds) )

        truths = np.array(truths)

        valid_results = []

        for c in range(num_classes):

            valid_results.append((c+1, np.sum(preds[:,c] * truths[:,c])/ np.sum(truths[:,c])))

        

        valid_results_df = pd.DataFrame(valid_results, columns=['Class', 'Valid_accuracy'])

        valid_results_df.to_csv('new_ValidationResults'+str(model_index+1)+'.csv', index=False)

        print(valid_results_df)



        del history, val_gen, train_aug_gen, valid_aug_gen, preds, truths, valid_results_df

        gc.collect()

        

        

    else:

        print('Load the pretrain weights '+str(model_index+1))

        model.load_weights(pretrain_weights_path[model_index])

        

    model_members.append(model)

    del model

    print('\n')
tta_aug = albu.Compose([

                    albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, shift_limit=0.10, p=1.0, border_mode=0, value=0)

                    ])
class tta_wrapper():



    def __init__(self,

                 model,

                 normal_generator,

                 aug_generator,

                 repeats = 1

                 ):

        '''

        model : model you trained on your original data

        normal_generator : generator for data without augmentation

        aug_generator : generator for data with augmentation

        repeats : how many times you want to use model to predict on augmentation data

        '''

        self.model = model

        self.normal_generator = normal_generator

        self.aug_generator = aug_generator

        self.repeats = repeats

    

        

    def predict(self):

        

        '''

        return : Averaging results of several different version of original test images

        '''



        batch_label = self.model.predict_generator( normal_generator,

                                                    workers=-1,

                                                    verbose=1)

        

        for idx in range(self.repeats):



            batch_label += self.model.predict_generator( aug_generator,

                                                         workers=-1,

                                                         verbose=1)

        batch_label /= (self.repeats+1)

        return batch_label
if submit == True:

    predictions = np.zeros((len(test_x), num_classes))



    for model_index, model in enumerate(model_members):

        print(str(model_index+1) + '_model predicting')

    

        normal_generator = InputGenerator(test_x,

                                          aug = albu.Compose([]),

                                          training=False,

                                          batch_size=250)

        

        aug_generator = InputGenerator(x=test_x,

                                       aug = tta_aug,

                                       training=False,

                                       batch_size=250)

    

        tta_model = tta_wrapper(model=model,

                                normal_generator=normal_generator,

                                aug_generator = aug_generator,

                                repeats=1)

    

        predictions += tta_model.predict()



    predictions = predictions / k_fold_split

    predictions = np.argmax(predictions, axis=1)

    submission = pd.DataFrame({'id' : test_indices, 'label':predictions})

    submission.to_csv('submission.csv', index = False)

    submission.head(5)
valid_labels = valid_df['label']

valid_datas = valid_df.drop(['label'],axis=1).values

valid_datas = valid_datas.reshape(-1,28,28,1)

valid_predictions = np.zeros((len(valid_datas), num_classes))



for model_index, model in enumerate(model_members):

    print(str(model_index+1) + '_model predicting')

    

    normal_generator = InputGenerator(x=valid_datas,

                                      aug = albu.Compose([]),

                                      training=False,

                                      batch_size=160)

        

    aug_generator = InputGenerator(x=valid_datas,

                                   aug = tta_aug,

                                   training=False,

                                   batch_size=160)

    

    tta_model = tta_wrapper(model=model,

                            normal_generator=normal_generator,

                            aug_generator = aug_generator,

                            repeats=1)

    

    valid_predictions += tta_model.predict()

    

valid_predictions = valid_predictions/k_fold_split

valid_predictions = np.argmax(valid_predictions,axis=1)


print('Real world validation accuracy : {}'.format(accuracy_score(valid_labels, valid_predictions)))
print(classification_report(valid_labels, valid_predictions))
plt.figure(figsize=(12,12))

confusion_mat = confusion_matrix(valid_labels, valid_predictions)

sn.heatmap(confusion_mat, annot=True, cmap='YlGnBu')

plt.title('Confusion matrix of Real World validation result')