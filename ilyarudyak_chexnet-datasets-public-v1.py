import pickle

import numpy as np

import pandas as pd

from sklearn.metrics import f1_score, confusion_matrix

from sklearn.model_selection import train_test_split

from pathlib import Path

import matplotlib.pyplot as plt

from functools import partial



import tensorflow as tf



AUTO = tf.data.experimental.AUTOTUNE
class Params:



    LEARNING_RATE = 0.001

    BATCH_SIZE = 32

    EPOCHS = 5



    #parameters for dataset

    CLASSES = [

        'Atelectasis',

        'Cardiomegaly',

        'Consolidation',

        'Edema',

        'Effusion',

        'Emphysema',

        'Fibrosis',

        'Hernia',

        'Infiltration',

        'Mass',

        # original label 'No Finding'

        # we don't include it in our labels

        # 'No_Finding',

        'Nodule',

        'Pleural_Thickening',

        'Pneumonia',

        'Pneumothorax'

    ]

    NO_FINDING = 'No_Finding'

    PNEUMONIA = 'Pneumonia'

    NOT_PNEUMONIA = 'not ' + PNEUMONIA

    TRAIN_SHUFFLE_BUFFER = 2048



    # parameters for the model

    IMAGE_SIZE = [224, 224]

    CHANNELS = 3



    # parameters for the trainer

    TRAIN_SPLIT = .8

    VAL_SPLIT = .1

    TEST_SPLIT = .1



    SEED = 42





params = Params()





class ChexnetUtils:



    @staticmethod

    def plot_sample_from_dataset(dataset, is_prep=True, is_pneumonia=True):

        dataset = dataset.take(9)

        plt.figure(figsize=(8, 8))

        subplot = 331

        for i, (image, label) in enumerate(dataset):

            plt.subplot(subplot)

            plt.axis('off')



            image = image.numpy()

            if is_prep: image = image / 2 + .5

            plt.imshow(image)



            label = label.numpy()

            if is_pneumonia: label = ChexnetUtils.get_label_pneumonia(label)

            else: label = ChexnetUtils.get_label_full(label)

            plt.title(label, fontsize=12)



            subplot += 1

            if i == 8: break

        # plt.tight_layout()

        plt.subplots_adjust(wspace=0.1, hspace=0.1)



    @staticmethod

    def get_label_pneumonia(label):

        return params.PNEUMONIA if label else params.NOT_PNEUMONIA



    @staticmethod

    def get_label_full(labels_arr):

        labels = [params.CLASSES[i] for i, j in enumerate(labels_arr) if j]

        if len(labels) == 0:

            return params.NO_FINDING

        return '|'.join(labels)



    @staticmethod

    def plot_sample_from_batch(batched_dataset, is_pneumonia=True):

        dataset = batched_dataset.take(1).unbatch()

        ChexnetUtils.plot_sample_from_dataset(dataset, is_pneumonia=is_pneumonia)



    @staticmethod

    def save_history(history, trainer):

        filename = trainer.history_file

        with open(filename, 'wb') as f:

            pickle.dump(history.history, f)



    @staticmethod

    def load_history(trainer):

        filename = trainer.history_file

        with open(filename, "rb") as f:

            history = pickle.load(f)

        return history



    @staticmethod

    def plot_metric(metric, trainer):

        history = ChexnetUtils.load_history(trainer)

        plt.plot(history[metric], label=metric)

        val_metric = f'val_{metric}'

        plt.plot(history[val_metric], label=val_metric)

        plt.legend()

        plt.title(metric)



    # @staticmethod

    # # TODO correct plotting of history with continued epochs

    # # TODO 2 history and weight files

    # def train_with_fine_tuning(metric='accuracy'):

    #     ft = FlowerTrainer()

    #     ft.train()

    #     FlowerUtils.plot_metric(metric=metric, trainer=ft)

    #     params.LEARNING_RATE /= 10

    #     params.EPOCHS = 20

    #     ft = FlowerTrainer(is_fine_tune=True)

    #     ft.train()

    #     FlowerUtils.plot_metric(metric=metric, trainer=ft)

    #     ft.evaluate()





class CheXNetDataset:

    """

    This class writes TFRecord files for NIH

    """



    def __init__(self, is_pneumonia=True):



        # directories and files

        self.DATA_DIR = Path('/kaggle/input/data')

        self.WORKING_DIR = Path('/kaggle/working')

        self.DF_LABELS_PATH = self.DATA_DIR / 'Data_Entry_2017.csv'

        self.DF_FILES_PATH = self.WORKING_DIR / 'df_files.csv'



        # dataframes columns

        self.FILE_PATH = 'file_path'

        self.PATIENT_ID = 'patient_id'

        self.SPLIT = 'split'

        self.LABELS = 'labels'

        self.LABELS_ARRAY = 'labels_array'

        self.LABEL_PNEUMONIA = 'label_pneumonia'

        self.PNEUMONIA = 'Pneumonia'



        # main dataframes and dictionaries

        self.df_original = None

        self.labels_dict = None

        self._get_labels_dict()

        self.df_files = None

        self._get_df_files()



        # dataframes for splits

        self.SPLIT_TRAIN = 'train'

        self.SPLIT_VAl = 'val'

        self.SPLIT_TEST = 'test'

        self.df_train = None

        self.df_val = None

        self.df_test = None

        self._get_patient_splits()



        # datasets

        self.is_pneumonia = is_pneumonia

        self.ds_train = None

        self.ds_val = None

        self.ds_test = None

        self._get_datasets()



    def _get_labels_dict(self):

        print(f'===> getting label_dict ... ')

        self.df_original = pd.read_csv(self.DF_LABELS_PATH)

        # change 'No Finding' to 'No_Finding'

        self.df_original .loc[self.df_original['Finding Labels'] == 'No Finding', 'Finding Labels'] = 'No_Finding'

        # get mapping of image name to its labels

        self.labels_dict = pd.Series(self.df_original['Finding Labels'].values,

                                     index=self.df_original['Image Index']).to_dict()

        print(f'===> getting label_dict DONE ')



    def _get_df_files(self):



        print(f'===> getting df_files ... ')



        # get list of paths for all image files

        if self.DF_FILES_PATH.is_file():

            files_str = list(pd.read_csv('df_files.csv').file_path.values)

        else:

            files_str = [str(fp) for fp in self.DATA_DIR.glob('**/**/*.png')]



        # create df that contain all image files

        self.df_files = pd.DataFrame(list(files_str), columns=[self.FILE_PATH], dtype='string')



        # add column with patient_id

        def get_patient(file_path): return str(file_path).split('/')[-1][:-8]

        self.df_files[self.PATIENT_ID] = self.df_files[self.FILE_PATH].apply(get_patient).astype('string')



        # add column with labels from original df provided in dataset

        def get_label(file_path): return self.labels_dict[file_path.split('/')[-1]]

        self.df_files[self.LABELS] = self.df_files[self.FILE_PATH].apply(get_label).astype('string')



        # add labels as array for full 14 disease problem

        def get_labels_as_array(labels):

            labels = labels.split('|')

            labels_arr = np.zeros(len(params.CLASSES))

            for label in labels:

                if label != params.NO_FINDING:

                    index = params.CLASSES.index(label)

                    labels_arr[index] = 1

            return labels_arr

        self.df_files[self.LABELS_ARRAY] = self.df_files[self.LABELS].apply(get_labels_as_array)



        # add label for pneumonia

        self.df_files[self.LABEL_PNEUMONIA] = self.df_files[self.LABELS].str.contains(self.PNEUMONIA).astype(int)



        # safe list of files to disc

        if not self.DF_FILES_PATH.is_file():

            self.df_files.file_path.to_csv(self.DF_FILES_PATH, index=False)



        print(f'===> getting df_files DONE ')



    def _get_patient_splits(self):



        print(f'===> getting patient_splits ... ')



        # get unique patient ids

        patients = self.df_files[self.PATIENT_ID].unique()



        # split patients using sklearn

        np.random.seed(params.SEED)

        patients_train, patients_val = train_test_split(list(patients),

                                                        train_size=params.TRAIN_SPLIT,

                                                        test_size=1-params.TRAIN_SPLIT,

                                                        random_state=params.SEED)

        patients_val, patients_test = train_test_split(patients_val,

                                                       train_size=.5,

                                                       test_size=.5,

                                                       random_state=params.SEED)

        patients_train, patients_val, patients_test = set(patients_train), set(patients_val), set(patients_test)



        # add split to main dataframe

        def get_split(patient_id):

            if patient_id in patients_train: return self.SPLIT_TRAIN

            elif patient_id in patients_val: return self.SPLIT_VAl

            elif patient_id in patients_test: return self.SPLIT_TEST

            else: raise ValueError('wrong patient_id')



        self.df_files[self.SPLIT] = self.df_files[self.PATIENT_ID].apply(get_split).astype('string')



        # get split dataframes

        self.df_train = self.df_files[self.df_files[self.SPLIT] == self.SPLIT_TRAIN]

        self.df_val = self.df_files[self.df_files[self.SPLIT] == self.SPLIT_VAl]

        self.df_test = self.df_files[self.df_files[self.SPLIT] == self.SPLIT_TEST]



        print(f'===> getting patient_splits DONE ')



    def _get_dataset(self, df, is_train=False):



        def _prepocess(file_path, label, _is_train=False):

            image = _decode_image(file_path)

            if _is_train:

                image = tf.image.random_flip_left_right(image)

            image = tf.cast(image, tf.float32)

            image = tf.keras.applications.xception.preprocess_input(image)

            return image, label



        def _decode_image(file_path):

            image_bytes = tf.io.read_file(file_path)

            image = tf.image.decode_jpeg(image_bytes, channels=3)

            image = tf.image.resize(image, params.IMAGE_SIZE)

            return image



        # load slices of dataframe into dataset

        if self.is_pneumonia: labels = df[self.LABEL_PNEUMONIA].values

        else: labels = np.stack(df[self.LABELS_ARRAY].values, axis=0)

        files = df[self.FILE_PATH].values

        list_ds = tf.data.Dataset.from_tensor_slices((files, labels))



        # preprocess dataset

        dataset = list_ds.map(partial(_prepocess, _is_train=is_train))



        if is_train:

            dataset = dataset.shuffle(params.TRAIN_SHUFFLE_BUFFER).repeat()



        dataset = dataset.batch(params.BATCH_SIZE).prefetch(AUTO)



        return dataset



    def _get_datasets(self):

        self.ds_train = self._get_dataset(df=self.df_train, is_train=True)

        self.ds_val = self._get_dataset(df=self.df_val)

        self.ds_test = self._get_dataset(df=self.df_test)
cds = CheXNetDataset()
cds.df_files.head()
cds.df_files.split.value_counts()
np.sum(cds.df_files.split.value_counts())
cds.df_train.shape, cds.df_val.shape, cds.df_test.shape
for image_batch, label_batch in cds.ds_train.take(1):

    print(image_batch.shape, label_batch.shape)
ChexnetUtils.plot_sample_from_batch(cds.ds_train)
cds = CheXNetDataset(is_pneumonia=False)
for image_batch, label_batch in cds.ds_train.take(1):

    print(image_batch.shape, label_batch.shape, label_batch[0])
ChexnetUtils.plot_sample_from_batch(cds.ds_train, is_pneumonia=False)