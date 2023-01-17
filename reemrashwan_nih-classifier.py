# to track experiments
!pip install comet_ml
from comet_ml import Experiment
### Loading Libraries
import numpy as np
import os
import pandas as pd

import tensorflow as tf
import sklearn
import matplotlib.pyplot as plt

from glob import glob
from tensorflow import keras

from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Dense, Dropout,Flatten, AveragePooling2D, GlobalAveragePooling2D, Concatenate
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.utils import plot_model
## Loading The Data

dataset_path = '/kaggle/input/data'
train_valid_images_path = os.path.join(dataset_path, "train_val_list.txt")
test_images_path = os.path.join(dataset_path, "test_list.txt")
dataset_df_path = os.path.join(dataset_path, 'Data_Entry_2017.csv')

with open(train_valid_images_path, 'r') as the_file:
    train_valid_images = the_file.read().splitlines()
    
with open(test_images_path, 'r') as the_file: 
    test_images = the_file.read().splitlines()

    
print(f"We have {len(train_valid_images)} training and validation images.")
print(f"We have {len(test_images)} testing images.")
train_valid_images[:3]
raw_dataset = pd.read_csv(dataset_df_path)
raw_dataset.info()

raw_dataset.head(10)
raw_dataset.info()

CLASSES = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

def preprocess_dataset_df(dataset, train_valid_images, test_images):
    def split_training_testing(df, train_valid_images, test_images):
        train_valid_df = df[df['Image Index'].isin(train_valid_images)]
        test_df = df[df['Image Index'].isin(test_images)]
        return train_valid_df, test_df
    
    def sample_data(df, num_samples):
    
        def get_rows_with_multiple_disease(df):
            return df[df['Finding Labels'].apply(lambda cell_value: True if '|' in cell_value else False)]

        def get_disease_samples(df, label, num_samples):
            subset_df = df[df['Finding Labels'] == label]
            return subset_df[:num_samples]

        # Sampling data
        multiple_disease = get_rows_with_multiple_disease(df)

        # empty df to concat in.
        single_disease = pd.DataFrame(columns = df.columns)

        for disease in CLASSES:
            disease_samples = get_disease_samples(df, disease, 2000)
            single_disease = pd.concat([disease_samples, single_disease])

        return pd.concat([multiple_disease, single_disease])

    
    def delete_extra_columns(dataset_df):
        needed_columns = ['Image Index', 'Finding Labels', 'Patient ID']
        dataset_df = dataset_df.drop(dataset_df.columns.difference(needed_columns), axis=1)
        return dataset_df
    
    def change_labels_as_list(dataset_df):
        dataset_df['Finding Labels'] = dataset_df['Finding Labels'].apply(lambda value: value.split('|'))
        return dataset_df
    
    def add_images_paths(dataset_df):
        data_image_paths = {os.path.basename(x): x for x in glob(os.path.join(dataset_path, 'images*', '*', '*.png'))}
        dataset_df['path'] = dataset_df['Image Index'].map(data_image_paths.get)
        return dataset_df
    

    # preprocess data
    new_dataset_df = delete_extra_columns(dataset)
    
    new_dataset_df, test_df = split_training_testing(new_dataset_df, train_valid_images, test_images)
    test_df = change_labels_as_list(test_df)
    test_df = add_images_paths(test_df)
    
    
    new_dataset_df = sample_data(new_dataset_df, 2000)
    new_dataset_df = change_labels_as_list(new_dataset_df)
    new_dataset_df = add_images_paths(new_dataset_df)
    return new_dataset_df.sort_index(), test_df

df, test_df = preprocess_dataset_df(raw_dataset, train_valid_images, test_images)
df.head()
df.info()
test_df.head()
from seaborn import countplot

# expand findings to simplify preprocessing
expanded_df = df[['Patient ID', 'Finding Labels']].explode('Finding Labels')

print("Number of samples in the dataset:", len(df))
print("Number of unique patients:", df['Patient ID'].nunique())
print("Number of classes: ", expanded_df['Finding Labels'].nunique())
print("Labels: ", expanded_df['Finding Labels'].unique())
# expand each list of Finding Labels column
# expanded_df = df[['Patient ID', 'Finding Labels']].explode('Finding Labels')
findings_count = expanded_df.groupby('Finding Labels')['Finding Labels'].count()
findings_count.sort_values(ascending=False)
def plot_disease_distribtion(df):
    df['Finding Labels'].value_counts().plot(kind="bar")
    
plot_disease_distribtion(df.explode('Finding Labels'))
def print_individual(df, disease):
    print(f"{disease}:", len(df[df['Finding Labels'].apply(lambda x: sorted(x) == [disease])]))
          
for disease in CLASSES:
    print_individual(df, disease)
# Samples of No Finding 
def get_samples_with_one_disease(df, disease):
    return df[df['Finding Labels'].apply(lambda x: sorted(x) == [disease])]

get_samples_with_one_disease(df, 'No Finding').sample(5)
# training parameters
BATCH_SIZE = 32
CLASS_MODE = 'categorical'
COLOR_MODE = 'rgb'
TARGET_SIZE = (256, 256)
EPOCHS = 10
SEED = 1337

print("length of training", len(df))
print("length of testing", len(test_df))
from sklearn.model_selection import train_test_split

train_patients_ids, valid_patients_ids = train_test_split(df['Patient ID'].unique(), test_size=0.15, random_state=42)
train_df = df[df['Patient ID'].isin(train_patients_ids)]
valid_df = df[df['Patient ID'].isin(valid_patients_ids)]
def print_ids_percentage(df1, df1_name, df2, df3):
    print(f"Percentage of {df1_name}: ", len(df1) / (len(df1) + len(df2) + len(df3)))
    
    
print("Common ids between sets:", len(set(train_df['Patient ID']) & set(valid_df['Patient ID']) & set(test_df['Patient ID'])))
print_ids_percentage(train_df, 'training', valid_df, test_df)
print_ids_percentage(valid_df, 'validation', train_df, test_df)
print_ids_percentage(test_df, 'testing', train_df, valid_df)
print("Data Size")
print("training:", len(train_df))
print("validation:", len(valid_df))
print("testing:", len(test_df))
plot_disease_distribtion(train_df.explode('Finding Labels'))
plot_disease_distribtion(valid_df.explode('Finding Labels'))
train_augmentation_parameters = dict(
#     preprocessing_function=preprocess_input,
    rescale=1/255,
#     rotation_range=10,
#     zoom_range=0.2,
    horizontal_flip=True
#     fill_mode='nearest',
#     brightness_range = [0.8, 1.2]
)

valid_augmentation_parameters = dict(
#     preprocessing_function=preprocess_input
    rescale=1/255
)

test_augmentation_parameters = dict(
#     preprocessing_function=preprocess_input
    rescale=1/255
)


train_consts = {
    'seed': SEED,
    'batch_size': BATCH_SIZE,
    'class_mode': CLASS_MODE,
    'color_mode': COLOR_MODE,
    'target_size': TARGET_SIZE,
    'classes': CLASSES,
    'shuffle': True
}

valid_consts = {
    'seed': SEED,
    'batch_size': BATCH_SIZE,
    'class_mode': CLASS_MODE,
    'color_mode': COLOR_MODE,
    'target_size': TARGET_SIZE, 
    'classes': CLASSES,
    'shuffle': False
}

test_consts = {
    'batch_size': 1,  # should be 1 in testing
    'class_mode': CLASS_MODE,
    'color_mode': COLOR_MODE,
    'target_size': TARGET_SIZE,
    'classes': CLASSES,
    'shuffle': False
}

# Using the training phase generators 
train_augmenter = ImageDataGenerator(**train_augmentation_parameters)
valid_augmenter = ImageDataGenerator(**valid_augmentation_parameters)
test_augmenter = ImageDataGenerator(**test_augmentation_parameters)
train_generator = train_augmenter.flow_from_dataframe(dataframe=train_df,
                             x_col='path',
                             y_col='Finding Labels',
                             **train_consts)

valid_generator = valid_augmenter.flow_from_dataframe(dataframe=valid_df,
                             x_col='path',
                             y_col='Finding Labels',
                             **valid_consts)

test_generator = test_augmenter.flow_from_dataframe(dataframe=test_df,
                             x_col='path',
                             y_col='Finding Labels',
                             **test_consts)
def get_class_weights(total_counts, class_positive_counts, multiply):
    """
    Calculate class_weight used in training
    Arguments:
    total_counts - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    multiply - int, positve weighting multiply
    use_class_balancing - boolean 
    Returns:
    class_weight - dict of dict, ex: {"Effusion": { 0: 0.01, 1: 0.99 }, ... }
    """
    def get_single_class_weight(pos_counts, total_counts):
        denominator = (total_counts - pos_counts) * multiply + pos_counts
        return {
            0: pos_counts / denominator,
            1: (denominator - pos_counts) / denominator,
        }

    class_names = list(class_positive_counts.keys())
#     print(class_positive_counts)
    label_counts = np.array(list(class_positive_counts.values()))
    class_weights = []
    for i, class_name in enumerate(class_names):
        class_weights.append(get_single_class_weight(label_counts[i], total_counts))

    return class_weights


def get_sample_counts(df):
    expanded_df = df[['Patient ID', 'Finding Labels']].explode('Finding Labels')
    findings_count = expanded_df.groupby('Finding Labels')['Finding Labels'].count()
    samples_count = df.shape[0]
    return samples_count, findings_count.to_dict()

# train_pos_counts is a dict with each class and number of occurences.
train_counts, train_pos_counts = get_sample_counts(train_df)
class_weights = get_class_weights(train_counts, train_pos_counts, multiply=1)

class_weights
# recall, precision, f1-score, AUROC for each class.
class Metrics(Callback):
    def __init__(self, val_data, *args, **kwargs):
        super().__init__()
        self.validation_data = val_data
        self.class_names = list(self.validation_data.class_indices.keys())
        self.reports = []
        self.aurocs = {}
        for c in self.class_names:
            self.aurocs[c] = []

    def on_epoch_end(self, epoch, logs={}):
        y_hat = np.asarray(self.model.predict(self.validation_data, verbose=1))
        y_hat = self.adjust_y_pred(y_hat)
        y_true = self.adjust_y_true()
        self.calculate_recall_precision_f1_score(y_true, y_hat, epoch)
        self.calculate_auroc(y_true, y_hat, epoch)
        return

    # Utility method
    def get(self, metrics, of_class):
        return [report[str(of_class)][metrics] for report in self.reports]

    def adjust_y_true(self):
        # self.validation_data
        val_trues = self.validation_data.classes
        y_true = sklearn.preprocessing.MultiLabelBinarizer().fit_transform(val_trues)
        return y_true

    def adjust_y_pred(self, predictions):
        y_pred = np.zeros(predictions.shape)
        y_pred[predictions > 0.5] = 1
        return y_pred

    def calculate_recall_precision_f1_score(self, y_true, y_hat, epoch):
        report = classification_report(y_true, y_hat,
                                       output_dict=True,
                                       target_names=self.class_names, zero_division=0)

        report_to_display = classification_report(y_true, y_hat,
                                                  target_names=self.class_names, zero_division=0)
        print('\n==========================')
        print(f"Epoch {epoch + 1} Metrics")
        print(report_to_display)

        self.reports.append(report)

    def calculate_auroc(self, y_true, y_hat, epoch):
        print("\n==========================")
        print(f"Epoch {epoch + 1} AUROCs.")
        current_auroc = []
        # calculate AUROC for each class.
        for i in range(len(self.class_names)):
            try:
                score = roc_auc_score(y_true[:, i], y_hat[:, i])
            except ValueError:
                score = 0
            # save class auroc score for this epoch
            self.aurocs[self.class_names[i]].append(score)
            # append classes AUROCs for the same epoch
            current_auroc.append(score)
            print(f'{i: >2}. {self.class_names[i]: >20}: {score}')
            # print(f"{i}. {self.class_names[i]}: {score}")

        # mean across classes
        mean_auroc = np.mean(current_auroc)
        print(f"\n{'': >5} Epoch {epoch + 1} Mean AUROC: {mean_auroc}")
        print("==========================")

        return

def create_dir(dirname):
    try:
        os.makedirs(dirname)
        print(f"Directory '{dirname}' created.") 
    except FileExistsError:
        print(f"Directory '{dirname}' already exists.")
models_dir = '/kaggle/working/models/'
model_name = 'NoImageNetNoAvgPoolingClassWeights'

model_path = os.path.join(models_dir, model_name)
best_model_path = os.path.join(model_path, 'best')
model_epochs_path = os.path.join(model_path, 'epochs')
model_logs_path = os.path.join(model_path, 'logs')

create_dir(models_dir)
create_dir(best_model_path)
create_dir(model_epochs_path)
create_dir(model_logs_path)
from tensorflow.keras.callbacks import *

multi_label_metrics = Metrics(valid_generator)

callbacks = [
    ModelCheckpoint(os.path.join(best_model_path, 'best'), monitor='val_loss',verbose=1, save_best_only=True),
    ModelCheckpoint(filepath=os.path.join(model_epochs_path, 'model.epoch{epoch:02d}-val_loss{val_loss:.2f}'), save_freq='epoch', period=10),
    ReduceLROnPlateau(factor=0.1, patience=1, min_lr=1e-8, verbose=1, cooldown=0),
    TensorBoard(model_logs_path), 
    CSVLogger(os.path.join(model_path, "model_history_log.csv"), append=True),
    multi_label_metrics
]
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input

encoder = DenseNet121(include_top=False, weights=None, input_shape=(*TARGET_SIZE, 3))

out_encoder = encoder.layers[426].output
# avg_conv = GlobalAveragePooling2D()(conv_base)

x = keras.layers.Flatten()(out_encoder)
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dense(256, activation='relu')(x)

output =  Dense(15, activation='sigmoid')(x)

model = Model(encoder.input, outputs=output)
# model.summary()

from tensorflow.keras.metrics import Accuracy, AUC, Precision, Recall
model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['accuracy'])
history = model.fit(
	x=train_generator,
    steps_per_epoch=len(train_generator),
	epochs=EPOCHS,
	validation_data=valid_generator,
    validation_steps=len(valid_generator),
    callbacks=callbacks,
    class_weight=class_weights
)
def plot_learning_metrics(history_model, to_plot):

    plt.plot(history_model.history[to_plot], label=to_plot)
    plt.plot(history_model.history['val_'+to_plot], label = 'val_' + to_plot)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
#     plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    plt.show()
plot_learning_metrics(history, 'loss')
plot_learning_metrics(history, 'accuracy')
# predictions = model.predict(valid_generator, steps=len(valid_generator), verbose=1)
# y_pred = np.zeros(predictions.shape)
# y_pred[predictions>0.5] = 1
# y_pred
# from sklearn.preprocessing import MultiLabelBinarizer
# val_trues = valid_generator.classes
# y_true = MultiLabelBinarizer().fit_transform(val_trues)
# y_true
# import sklearn
# print(sklearn.metrics.classification_report(y_true, y_pred, target_names=valid_generator.class_indices.keys()))
# sklearn.metrics.roc_auc_score(y_true, y_pred)
# import libraries
# from sklearn.metrics import roc_curve, auc

# # create plot
# fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
# for (i, label) in enumerate(dummy_labels):
#     fpr, tpr, thresholds = roc_curve(test_Y[:,i].astype(int), quick_model_predictions[:,i])
#     c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (label, auc(fpr, tpr)))

# # Set labels for plot
# c_ax.legend()
# c_ax.set_xlabel('False Positive Rate')
# c_ax.set_ylabel('True Positive Rate')
# fig.savefig('quick_trained_model.png')