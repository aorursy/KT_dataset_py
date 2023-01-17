import pandas_profiling

from pandas_profiling import ProfileReport

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import random

import glob

import tensorflow as tf



seed = 16

tf.random.set_seed(seed)

np.random.seed(seed)

random.seed(seed)
from pathlib import Path

from PIL import Image



def show_image(path, extra_title = ''):

    img = Image.open(path)

    img = img.convert("RGB")

    img_size = str(img.size)

    extra_title = str(extra_title)

    plt.title('Label: {}, Image Size: {}'.format(extra_title, img_size))

    plt.imshow(img)

    plt.show()

    



def make_a_tuple(row):

    row = row.replace('|', ',')

    row = row.split(",")

    row = tuple(row)

    return row



def read_data(img_dir = '/kaggle/input/data/',

              df_dir = '/kaggle/input/data/Data_Entry_2017.csv'):



    sample_df = pd.read_csv(df_dir)

    print('Original columns: {}'.format(sample_df.columns.tolist()))

    sample_df['freq_customer'] = sample_df.groupby('Patient ID')['Follow-up #'].transform(sum)



    sample_df = sample_df[['Image Index', 'Finding Labels', 'View Position', 'Patient ID', 'freq_customer']]

    sample_df.columns = ['filename','class', 'view', 'patient_id', 'freq_customer']

    sample_df.head()

    print('New columns: {}'.format(sample_df.columns.tolist()))

    print('Rows: {}'.format(len(sample_df)))

    

    # replace class names with tuple

    sample_df['class'] = sample_df['class'].apply(make_a_tuple)

    

    # replace relative with an absolute path

    absolute_paths = glob.glob('/kaggle/input/data/*/*/*')

    relative_paths = [a.split('/')[-1] for a in absolute_paths]



    path_mapping = pd.DataFrame({'path': absolute_paths, 

                                 'filename' : relative_paths})



    sample_df = pd.merge(sample_df, path_mapping)

    sample_df = sample_df.drop(columns = 'filename')

    sample_df = sample_df.rename(columns= {'path': 'filename'})

    

    sample_df['len'] = [len(a) for a in sample_df['class']]

    



    return sample_df
def take_smaller_sample(sample_df, frac, seed = 16):

    sample_df = sample_df.sample(frac = frac, random_state = seed)

    print('Sample rows {}'.format(len(sample_df)))

    return sample_df
from collections import Counter

from itertools import chain

import matplotlib.pyplot as plt



def count_classes(sample_df, df_type = 'train', print_info = True):

    if print_info:

        print('\ntotal {}_df rows: {}'.format(df_type, len(sample_df)))

        

    flat = get_value_counts(sample_df['class'], print_info)

    total_counts = Counter(chain.from_iterable(flat))

    total_counts = total_counts.most_common()

    

    max_class = total_counts[0][1]

    min_class = total_counts[-1][1]

    imbalance = max_class / min_class

    

    if print_info:

        print('class imbalance of {0:.0f}x'.format(imbalance))

    

    return total_counts

    

def get_value_counts(df_column, print_info = True):

    flat= df_column.tolist()

    

    flat_clean = [f for f in flat if f != ('No Finding',)]

    mixed = [f for f in flat_clean if len(f) > 1]

    pct_mixed = len(mixed)/len(flat_clean) * 100

    

    if print_info:

        print('mixed labels are {0:.2f}%'.format(pct_mixed))

    

    return flat



def get_class_bar(sample_df):

    total_counts = count_classes(sample_df)



    fig, ax = plt.subplots(figsize=(7, 10))

    illness = [i[0] for i in total_counts] 

    counts = [i[1] for i in total_counts] 

    

    y_pos = np.arange(len(illness))

    ax.barh(y_pos, counts, align='center')

    ax.set_yticks(y_pos)

    ax.set_yticklabels(illness)

    ax.invert_yaxis()  

    ax.set_xlabel('Number of images')

    ax.set_title('How many images do we have per class?')

    ax.set_xticks(np.arange(0, max(counts), np.mean(counts)/1.5))



    plt.show()
from sklearn.model_selection import train_test_split



def get_len(df,split):

    print("{} rows: {}".format(split, len(df)))

    

def split_data(sample_df, seed, test_size, validation_size):

    

    train_size = 1 - test_size - validation_size

    

    train_df, test_df, validate_df = np.split(sample_df.sample(frac=1), [int(train_size*len(sample_df)), int((1-validation_size)*len(sample_df))])

    

    train_share = len(train_df)/len(sample_df) * 100

    test_share = len(test_df)/len(sample_df) * 100

    valid_share = len(validate_df)/len(sample_df) * 100



    

    print("{}% train set, {} rows".format(train_share, len(train_df)))

    print("{}% test set, {} rows".format(test_share, len(test_df)))

    print("{}% validation set, {} rows".format(valid_share, len(validate_df)))

    

    train_df.reset_index(inplace = True, drop= True)

    test_df.reset_index(inplace = True, drop= True)

    validate_df.reset_index(inplace = True, drop = True)

    

    count_classes(test_df, 'test', True)

    count_classes(validate_df, 'validate', True)

    count_classes(train_df, 'train', True)



    return train_df, test_df, validate_df

# only PA view to start



def filter_view(sample_df, view):

    sample_df = sample_df[sample_df['view'] ==  view]

    print('Left only {}'.format(sample_df['view'].value_counts().index))

    print('Rows: {}'.format(len(sample_df)))

    return sample_df
# get rid of most of no finding

def downsample_class(label, df, max_size, only_single_labels):

    """ Downsample one clsas,

        choice between downsampling

        images which only have one label

        vs labels which have multiple labels"""



    if only_single_labels:

        contains = [True if a == (label,) else False for a in df['class']]



    else:

        contains = [True if (label in a)  else False for a in df['class']]



    not_contains = [not i for i in contains]



    if len(df[contains]) > max_size:

        sample = df[contains].sample(max_size)

        df = pd.concat([df[not_contains], sample])



    return df



def undersample(df, undersample, median_factor = 2, only_single_labels = False, print_info = False):

    """ Undersample all classes that are above

         median_factor, e.g. 2x of the median"""

    

    if undersample:

        counts = count_classes(df, 'train', print_info)

        median = np.median([count for key, count in counts])

        max_size = int(median * median_factor)

        print('\nundersampling largest classes to {}'.format(max_size))





        names = [c[0] for c in counts]



        for name in names: 

            df = downsample_class(name, df, max_size, only_single_labels)



        count_classes(df)

    else:

        print('not undersampling')

    return df
def show_images(df_slice):

    for i, row in df_slice.iterrows():

        show_image(row['filename'], extra_title = row['class'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import shutil



def upsample_class(df, 

                   min_size,

                   image_size,

                   class_name = 'Hernia', 

                   save_dir = '/kaggle/working/',

                   batch_size = 64,

                   color_mode = 'rgb'):

        """ Function to upsample the amoung of images for classes which 

            have little amount of images. To avoid directly copying the images

            we augment them 

            """ 

        counts = count_classes(df, 'train', False)

        counts = dict(counts)

        n_current_images = counts[class_name]

        

        if n_current_images < min_size:

            images_to_add = min_size - n_current_images



            append_df  = {'class' : [], 

                          'filename': []}



            augment_generator = ImageDataGenerator(rescale=None,

                                                   horizontal_flip=True,

                                                   width_shift_range=0.15,

                                                   rotation_range= 15,

                                                   height_shift_range=0.15,

                                                   shear_range=0.15, 

                                                   zoom_range=0.15,

                                                   brightness_range=[0.7, 1.0])



            df['filter'] = [True if class_name in i else False for i in df['class']]

            class_df = df[df['filter'] == True]



            save_dir =  os.path.join(save_dir,class_name)



            if os.path.exists(save_dir):

                shutil.rmtree(save_dir, ignore_errors=False)



            os.makedirs(save_dir, exist_ok = True)



            imageGen = augment_generator.flow_from_dataframe(class_df,

                                                             batch_size = 1,

                                                             target_size = image_size,

                                                             save_to_dir = save_dir,

                                                             shuffle = False,

                                                             color_mode= color_mode)



            # check if already did that

            n_already = len(glob.glob(save_dir + '/*'))



            print('{} is found in {} images'.format(class_name, len(class_df)))

#             print('The folder {} has {} images'.format(save_dir, n_already))

            print('Adding {} images'.format(images_to_add))



            if n_already < images_to_add:

                total = n_already

                class_index = 0



                # iterate over images to generate more data

                for image in imageGen:

                    if class_index >= len(class_df):

                        # preserve the original classes, not just single class

                        class_index = class_index - len(class_df) 



                    # append the class and reset the state if iloc goes beyond the len of df

                    append_df['class'].append(class_df.iloc[class_index]['class'])



                    total += 1

                    class_index += 1



                    if total == images_to_add:

                        break



                files = glob.glob(save_dir + '/*')

                files.sort(key=os.path.getmtime)



                append_df['filename'] = [file for file in files]

                append_df = pd.DataFrame(append_df) 



                df = pd.concat([df, append_df])

                df = df.drop(columns = 'filter')

    

        return df
import shutil



def upsample(df, upsample, image_size, color_mode, median_factor = 0.5):

    if upsample:

        counts = count_classes(df,'train', False) 

        median = np.median([count for key, count in counts])

        min_size = int(median * median_factor)

        print('\nupsampling smallest classes to {}'.format(min_size))

        

        names = [c[0] for c in counts]



        for name in names: 

            df = upsample_class(df = df, 

                                class_name = name,

                                min_size = min_size,

                                image_size = image_size,

                                color_mode = color_mode)

            

    else:

        print('not upsampling')



    return df
# data generators

# from tensorflow.keras.applications.inception_v3 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator



def get_image_generators(train_df,

                         validate_df,

                         rescale,

                         do_data_augmentation,

                         preprocessing_function,

                         target_size,

                         batch_size,

                         class_mode,

                         seed,

                         color_mode):



    valid_datagen = ImageDataGenerator(

                                       rescale=rescale,

                                       preprocessing_function = preprocessing_function

    )



    if do_data_augmentation:

        train_datagen = ImageDataGenerator(

                                           rescale=rescale,

                                           horizontal_flip=True,

                                           width_shift_range=0.05,

                                           rotation_range= 5,

                                           height_shift_range=0.05,

                                           shear_range=0.05, 

                                           zoom_range=0.05,

                                           brightness_range=[0.8, 1.0],

                                           preprocessing_function=preprocessing_function)

    else:

        train_datagen = valid_datagen

        





    dataflow_kwargs = dict(target_size = target_size, batch_size=batch_size,

                           interpolation="bilinear", class_mode=class_mode, color_mode = color_mode)





    train_dataflow = train_datagen.flow_from_dataframe(train_df,

                                                       shuffle=True,

                                                       y_col = 'class',

                                                       x_col="filename",

                                                       seed=seed,

                                                       **dataflow_kwargs)



    valid_dataflow = valid_datagen.flow_from_dataframe(validate_df,

                                                       y_col = 'class',

                                                       x_col="filename",

                                                       shuffle=False, 

                                                       seed=seed,

                                                       **dataflow_kwargs)

    

    return train_dataflow, valid_dataflow, dataflow_kwargs, valid_datagen

import tensorflow_hub as hub

import tensorflow as tf



# basics



def build_model(train_dataflow,

                params,

                image_size,

                epochs, 

                dropout,

                model,

                activation,

                n_channels

                ):



    num_classes = len(train_dataflow.class_indices)

    image_size = params['image_size']

    

    hub_models = ["mobilenet_v2_100_224", "inception_v3"]

    keras_applications = ['ResNet101V2', 'InceptionV3']

    

    input_shape = image_size + (n_channels,)



        

    if model in hub_models:

        choice = hub_models.index(model)

        module_handle = "https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(hub_models[choice])



        print("Using {} with input size {}".format(module_handle, params['image_size']))

        print("Building model with", module_handle)

        print('up from here')





        base_model = [

                      tf.keras.layers.InputLayer(input_shape=params['image_size'] + (n_channels,)),

                      hub.KerasLayer(module_handle, trainable=params['do_fine_tuning']),

                      tf.keras.layers.Dropout(rate=dropout)

                     ]

    

    elif model in keras_applications:

        kwargs = dict(weights=params['weights'], include_top = False,

                      input_shape=input_shape, pooling='max')



        if model == 'InceptionV3':

            base_model = [tf.keras.layers.InputLayer(input_shape=params['image_size'] + (n_channels,)),

                          tf.keras.applications.InceptionV3(**kwargs),

                          tf.keras.layers.Dropout(rate=dropout)

                         ]

        elif model == 'ResNet101V2':

            base_model = [tf.keras.layers.InputLayer(input_shape=params['image_size'] + (n_channels,)),

                          tf.keras.applications.ResNet101V2(**kwargs),

                          tf.keras.layers.Dropout(rate=dropout)

                         ]

    

    else:

        raise ValueError("Model should be of type Keras application or TF Hub")

    

    last_layer = [tf.keras.layers.Dense(num_classes,

                                        kernel_regularizer=tf.keras.regularizers.l2(params['l2']),

                                        activation= activation)

                 ]

    

    model = tf.keras.Sequential(base_model + last_layer)

    model.build((None,) + input_shape)

    print(model.summary())



    return model
def compile_model(model, params):

    if params['optimizer'] == 'Adam':

        optimizer = tf.keras.optimizers.Adam(lr=params['lr'])

    

    elif params['optimizer'] == 'SGD':

        optimizer=tf.keras.optimizers.SGD(lr=params['lr'], momentum= params['momentum'])



    else:

        raise ValueError('Optimizer should be of type [SGD, Adam]')



    model.compile(optimizer=optimizer,

                  loss=params['loss'],

                  metrics=params['metrics'])

    

    return model
def train(model, 

          train_dataflow, 

          valid_dataflow,

          params):

    



    print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")



    steps_per_epoch = train_dataflow.samples // train_dataflow.batch_size

    validation_steps = valid_dataflow.samples // valid_dataflow.batch_size



    hist = model.fit(

                    train_dataflow,

                    epochs=params['epochs'], 

                    steps_per_epoch=steps_per_epoch,

                    validation_data=valid_dataflow,

                    validation_steps=validation_steps,

                    verbose = 2).history

    

    return model, hist
from sklearn.metrics import multilabel_confusion_matrix

import sys

from sklearn.metrics import (f1_score,  precision_score,  recall_score)



def get_results_per_class(test_dataflow, y_true, preds, threshold = 0.5):

    label_to_class = {v: k for k, v in test_dataflow.class_indices.items()}



    results_df = {'class': [],

                  'precision':[],

                  'recall': []}

    

    preds = preds > threshold

    

    for i, conf_matrix in enumerate(multilabel_confusion_matrix(y_true, preds)):

            tn, fp, fn, tp = conf_matrix.ravel()

            f1 = 2 * tp / (2 * tp + fp + fn + sys.float_info.epsilon)

            recall = tp / (tp + fn + sys.float_info.epsilon)

            precision = tp / (tp + fp + sys.float_info.epsilon)

            results_df['class'].append(label_to_class[i])

            results_df['precision'].append(precision)

            results_df['recall'].append(recall)



    results_df = pd.DataFrame(results_df)

    

    return results_df
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.metrics import average_precision_score, roc_auc_score



def test_model(test_df, model, valid_datagen, dataflow_kwargs, seed = 16):

    print('\ntesting the model...')

    

    test_dataflow = valid_datagen.flow_from_dataframe(test_df,

                                                      shuffle=False, 

                                                      seed=seed,

                                                      **dataflow_kwargs) 



    results = model.evaluate(test_dataflow)



    multilabelbinarizer = MultiLabelBinarizer()

    y_true = multilabelbinarizer.fit_transform(test_df['class'])



    preds = model.predict(test_dataflow)

    auc_score = results[1]

    ap_score = average_precision_score(y_true = y_true, 

                                       y_score = preds, 

                                       average = 'macro')    

    

    roc_score = roc_auc_score(y_true, preds)

    

    print('auc: {}, roc: {}, ap: {}'.format(auc_score, roc_score,  ap_score))

    results_df = get_results_per_class(test_dataflow, y_true, preds)

    

    metrics = {'auc': auc_score,

               'ap': ap_score,

               'roc': roc_score}



    artifacts = {'df': results_df}



    return metrics, artifacts
try:

    import mlflow



except:

    !pip install mlflow --quiet 

    import mlflow



from datetime import datetime



def log_mlflow(experiment_name,

               run_name,

               params,

               metrics,

               artifacts,

               hist):

    

    time = str(datetime.utcnow()).replace('.','_').replace(' ', '_')    

    mlflow_ui_path = os.path.normpath(os.path.join('file:/' + '/kaggle/working/', 'mlruns'))

    mlflow.set_tracking_uri(mlflow_ui_path)

    check_exists = mlflow.get_experiment_by_name(experiment_name)



    if check_exists is None:

        mlflow.create_experiment(experiment_name)



    mlflow.set_experiment(experiment_name)

    

    """ Logs results to MLFlow runs"""

    with mlflow.start_run(run_name=run_name) as run:

        for param in params:

            mlflow.log_param(param, params[param])



        for metric in metrics:

            mlflow.log_metric(metric,metrics[metric])

        

        stats_pathname = '/kaggle/working/stats_{}.csv'.format(time)

        artifacts['df'].to_csv(stats_pathname)

        mlflow.log_artifact(stats_pathname)

        

        runID = run.info.run_uuid

        experimentID = run.info.experiment_id

        print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))

        

    experiment = mlflow.get_experiment_by_name(experiment_name)

    experiment_id = experiment.experiment_id

    df_runs = mlflow.search_runs(experiment_ids=[experiment_id])

    df_runs.to_csv('/kaggle/working/mlruns.csv') 

    save_history(hist, time)



    

def save_history(hist, time):

    if "loss" in hist.keys():

        plt.figure()

        plt.ylabel("Loss (training and validation)")

        plt.xlabel("Training Epochs")

        plt.ylim([0, 2])

        plt.plot(hist["loss"])

        plt.plot(hist["val_loss"])

        plt.legend(['train', 'test'])

        plt.savefig('/kaggle/working/loss_{}.png'.format(time))
def run_training(params):

    print(params)



    sample_df = read_data()

    sample_df = take_smaller_sample(sample_df, frac = params['frac'])



    if params['view'] != 'all':

        sample_df = filter_view(sample_df,  params['view'])





    train_df, test_df, validate_df = split_data(sample_df, 

                                                  seed = 16, 

                                                  test_size = params['test_size'],

                                                  validation_size = params['validation_size'])



    train_df = undersample(train_df, 

                           undersample = params['undersample'],

                           median_factor = params['undersample_median_factor'],

                           only_single_labels = params['undersample_only_single'])



    train_df = upsample(train_df, 

                        upsample = params['upsample'], 

                        median_factor = params['upsample_median_factor'],

                        image_size = params['image_size'],

                        color_mode = params['color_mode'])

    

    print('Classes train set: {}'.format(count_classes(train_df)))  

    print('Classes test set: {}'.format(count_classes(test_df, print_info = False)))  



    train_dataflow, valid_dataflow, dataflow_kwargs, valid_datagen =  get_image_generators(train_df = train_df,

                                                                                           validate_df = validate_df,

                                                                                           rescale= params['rescale'],

                                                                                           do_data_augmentation = params['do_augmentation'],

                                                                                           preprocessing_function = params['preprocessing_function'],

                                                                                           target_size = params['image_size'],

                                                                                           batch_size = params['batch_size'],

                                                                                           class_mode = params['class_mode'],

                                                                                           seed = params['seed'],

                                                                                           color_mode = params['color_mode'])

    

    model = build_model(train_dataflow = train_dataflow,

                        params = params,

                        image_size = params['image_size'],

                        epochs = params['epochs'],

                        dropout = params['dropout'],

                        model = params['model'],

                        activation = params['activation'],

                        n_channels = params['n_channels'])

                        

    model = compile_model(model = model, 

                          params=params)

        

    model, hist = train(model = model, 

                        train_dataflow = train_dataflow, 

                        valid_dataflow = valid_dataflow,

                        params = params)

    

    metrics, artifacts = test_model(test_df = test_df, 

                                     model= model, 

                                     dataflow_kwargs = dataflow_kwargs, 

                                     valid_datagen = valid_datagen,

                                     seed = params['seed'])

    

    df_runs = log_mlflow(experiment_name = 'X-ray', 

                         run_name = 'First iteration runs', 

                         params= params, 

                         metrics=metrics,

                         artifacts =artifacts,

                         hist=hist)

    

    
params = {

    

    # data

    'frac': 1,

    'test_size': 0.2,

    'validation_size': 0.1,

    'view': 'all',

    'image_size': (299,299),

    'color_mode': 'rgb',

    'n_channels': 3,

    'do_augmentation': True,

    'rescale': 1./ 255,

    'preprocessing_function': None,

    'class_mode': 'categorical',

    

    # class balance

    'upsample': True,

    'upsample_median_factor': 0.5,

    'undersample': True,

    'undersample_median_factor': 2,

    'undersample_only_single': False,

    

    # hyper params

    'batch_size': 64,

    'lr': 0.005,

    'optimizer': 'SGD',

    'epochs' : 8,

    'dropout': 0.0,

    'smoothing': 0,

    'l2' : 0.0001,

    'momentum' : 0.9,

    

    # model

    'model' : 'inception_v3',

    'do_fine_tuning' : True,

    'weights':'imagenet',

    

    # metrics

    'loss' : tf.keras.losses.CategoricalCrossentropy(),

    'metrics' : [tf.keras.metrics.AUC(multi_label= True), 

                 tf.keras.metrics.Recall()],

    

    'activation': 'sigmoid',

    'get_pandas_profiling': False,

    'seed': 16

}
import time



start = time.time()



def run_experiment(params, changes = None):

    try:

        if changes is not None:

            for k,v in changes.items():

                params[k] = v



            run_training(params)

        

    except Exception as e:

        print(e)

        print('failure')

        

run_experiment(params, changes = {'epochs': 8})
# mode block: model types, pre-rpocessing

# best model - fine tune, unfreeze, freeze

# greyscaling



# to-do:

# retrainable ['all','some','none']

# last layer ['dense', 'no dense']

# drop negative ['yes','no']

# negative as a label/without

# regularization ['l2', 'no l2']



# doing:

# see training loss, apply regularization

# check nodule and mass compared to atelectasis



# done:

# balancing (no/down/full +) - full wins

# loss function(cosine/binary/cross +) - cross entropy wins

# data augment ['yes', 'no'] - yes wins

# models ['inception', 'mobilenet'] - inception wins

# fine tune ['true', 'false'] - true wins

# image size ['299', '224'] - 299 wins

# models ['resnet50','densenet'] - tf hub wins over keras applications

# n_channels ['3','1'] - 3 channels wins

# greyscale ['yes','no'] - color wins

# full size vs sample - full size wins



# not possible:

# deduplicate images ['yes','no']

# view ('PA', 'AP', 'all') - same, did not try

# threshold ['0.3', '0.5'] - does not matter for training (only tseting)

import tensorflow as tf

import numpy as np

from sklearn.metrics import average_precision_score, roc_auc_score, multilabel_confusion_matrix



true_label = np.array(([1, 1, 1, 1, 1, 1, 1], 

                      [0, 0, 1, 0, 0, 0, 1],

                      [0, 0, 0, 0, 0, 0, 0],

                      [1, 0, 1, 1, 0, 0, 0]))



pred_label = np.array(([1, 1, 1, 1, 1, 1, 1], 

                      [0, 0, 1, 0, 0, 0, 1],

                      [0, 0, 0., 0, 0, 0, 0],

                      [0., 0, 0.4, 0., 0, 0, 0]))



# Accuracy

m = tf.keras.metrics.AUC(num_thresholds = 100, multi_label = True)

m.update_state(y_true = true_label, y_pred = pred_label)

print(m.result().numpy().round(2))



# Accuracy

m = tf.keras.metrics.AUC(num_thresholds = 100, multi_label = False)

m.update_state(y_true = true_label, y_pred = pred_label)

print(m.result().numpy().round(2))



ap_score = average_precision_score(y_true = true_label, 

                                   y_score = pred_label, 

                                   average = 'macro')    



print(ap_score.round(2))



roc_score = roc_auc_score(true_label, pred_label)

print(roc_score.round(2))
threshold = 0.5

pred_label = pred_label > threshold

pred_label
# Accuracy

m = tf.keras.metrics.AUC(num_thresholds = 100, multi_label = True)

m.update_state(y_true = true_label, y_pred = pred_label)

print(m.result().numpy().round(2))



# Accuracy

m = tf.keras.metrics.AUC(num_thresholds = 100, multi_label = False)

m.update_state(y_true = true_label, y_pred = pred_label)

print(m.result().numpy().round(2))



ap_score = average_precision_score(y_true = true_label, 

                                   y_score = pred_label, 

                                   average = 'macro')    



print(ap_score.round(2))



roc_score = roc_auc_score(true_label, pred_label)

print(roc_score.round(2))