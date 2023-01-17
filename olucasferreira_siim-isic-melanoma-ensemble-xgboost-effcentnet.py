import os

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import missingno as msno

import pydicom as dcm



import warnings

warnings.filterwarnings('ignore')





plt.style.use('ggplot')
PATH = "/kaggle/input/siim-isic-melanoma-classification/"



train_df = pd.read_csv(os.path.join(PATH, 'train.csv'))

test_df = pd.read_csv(os.path.join(PATH, 'test.csv'))

sample_submission_df = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
print(f"Train shape: {train_df.shape}")

print(f"Test shape: {test_df.shape}")

print(f"Sample submission shape: {sample_submission_df.shape}")
# Change columns names

new_names = ['image_name', 'ID', 'sex', 'age', 'anatomy', 'diagnosis', 'benign_malignant', 'target']

train_df.columns = new_names

test_df.columns = new_names[:5]
train_df.head()
test_df.head()
# Visualizing the missing values

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))



msno.matrix(train_df, ax=ax1, fontsize=10)

msno.matrix(test_df, ax=ax2, fontsize=10)



ax1.set_title('Train Missing Values Map', fontsize=15)

ax2.set_title('Test Missing Values Map', fontsize=15)
print(train_df.shape)

train_df.isnull().sum()
print(test_df.shape)

test_df.isnull().sum()
# Unique IDs

print(f"The total patient IDs are {train_df.ID.count()}, from those the unique IDs are {train_df.ID.value_counts().shape[0]}.")
# Number of images by ID

patients_train = train_df.groupby('ID')['image_name'].count().reset_index()

patients_test = test_df.groupby('ID')['image_name'].count().reset_index()



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

sns.distplot(patients_train.image_name, kde=False, bins=50, ax=ax1)

sns.distplot(patients_test.image_name, kde=False, bins=50, ax=ax2)



ax1.set_title('Images by Patient Distribution - Train')

ax2.set_title('Images by Patient Distribution - Test')
# Diagnosis and target

plt.figure(figsize=(8, 5))

sns.countplot(data=train_df, x='sex')

plt.title('Gender Distribuition')
sex_mode = train_df.sex.mode()[0]

train_df.sex.fillna(sex_mode, inplace=True)

print('Mode:', sex_mode)
# Age Variable

plt.figure(figsize=(10, 5))

sns.distplot(train_df.age)

plt.title('Age Distribuition')
train_df.age.describe()
# Filling the age variable with the median: 50 years.

age_median = train_df.age.median()

train_df.age.fillna(age_median, inplace=True)

print('Median:', age_median)
# Age distribution and sex

plt.figure(figsize=(15, 5))

sns.distplot(train_df[train_df.sex == 'male']['age'])

sns.distplot(train_df[train_df.sex == 'female']['age'])

plt.title('Age Distribution by Gender')
plt.figure(figsize=(15, 5))

sns.countplot(data=train_df, x='anatomy')

plt.title('Scanned Body Parts')
train_df.anatomy.value_counts(dropna=False)
# Checking the 'benign_malignant' variable of NaN values in 'anatomy'

train_df[train_df.anatomy.isnull()]['benign_malignant'].value_counts()
train_df.anatomy.fillna('torso', inplace=True)
# Checking the test data

test_df.isnull().sum()
# Age variable of NaN values in 'anatomy' variable 

test_df[test_df.anatomy.isnull()]['age'].value_counts()
age_nan = test_df[test_df['age'] == 70]['anatomy'].value_counts().reset_index()['index'][0]

test_df.anatomy.fillna(age_nan, inplace=True)
# Genders by Anatomy

plt.figure(figsize=(15, 5))

sns.countplot(data=train_df, x='anatomy', hue='sex')

plt.title('Anatomy Distribution by Genders')
plt.figure(figsize=(15, 5))

sns.countplot(data=train_df, x='diagnosis')

plt.title('Diagnosis Distribuiton')
# Diagnosis and target

plt.figure(figsize=(15, 5))

sns.countplot(data=train_df, x='benign_malignant', hue='diagnosis')

plt.title('Diagnosis Distribution by Target')
# Diagnosis and sex

plt.figure(figsize=(15, 5))

sns.countplot(data=train_df, x='diagnosis', hue='sex')

plt.title('Diagnosis Distribution by Genders')
# Target

train_df.target.value_counts()
sns.countplot(data=train_df, x='benign_malignant')

plt.title('Target Distribuition')
# Genders by target

plt.figure(figsize=(10, 5))

sns.countplot(data=train_df, x='benign_malignant', hue='sex')

plt.title('Genders Distribuition by Target')
# Anatomy by target

plt.figure(figsize=(15, 5))

sns.countplot(data=train_df, x='anatomy', hue='benign_malignant')

plt.title('Anatomy Distribuition by Target')
def show_dicom_images(data):

    img_data = list(data.T.to_dict().values())

    f, ax = plt.subplots(3,3, figsize=(16,18))

    for i,data_row in enumerate(img_data):

        patientImage = data_row['image_name']+'.dcm'

        imagePath = os.path.join(PATH,"train/",patientImage)

        data_row_img_data = dcm.read_file(imagePath)

        modality = data_row_img_data.Modality

        age = data_row_img_data.PatientAge

        sex = data_row_img_data.PatientSex

        data_row_img = dcm.dcmread(imagePath)

        ax[i//3, i%3].imshow(data_row_img.pixel_array, cmap=plt.cm.gray) 

        ax[i//3, i%3].axis('off')

        ax[i//3, i%3].set_title(f"ID: {data_row['image_name']}\nAge: {age} Sex: {sex}\nDiagnosis: {data_row['diagnosis']}")

    plt.show()
show_dicom_images(train_df[train_df.target == 0].sample(9))
show_dicom_images(train_df[train_df.target == 1].sample(9))
def extract_DICOM_attributes(folder):

    images = list(os.listdir(os.path.join(PATH, folder)))

    df = pd.DataFrame()

    for image in images:

        image_name = image.split(".")[0]

        dicom_file_path = os.path.join(PATH,folder,image)

        dicom_file_dataset = dcm.read_file(dicom_file_path)

        study_date = dicom_file_dataset.StudyDate

        modality = dicom_file_dataset.Modality

        age = dicom_file_dataset.PatientAge

        sex = dicom_file_dataset.PatientSex

        body_part_examined = dicom_file_dataset.BodyPartExamined

        patient_orientation = dicom_file_dataset.PatientOrientation

        photometric_interpretation = dicom_file_dataset.PhotometricInterpretation

        rows = dicom_file_dataset.Rows

        columns = dicom_file_dataset.Columns



        df = df.append(pd.DataFrame({'image_name': image_name, 

                        'dcm_modality': modality,'dcm_study_date':study_date, 'dcm_age': age, 'dcm_sex': sex,

                        'dcm_body_part_examined': body_part_examined,'dcm_patient_orientation': patient_orientation,

                        'dcm_photometric_interpretation': photometric_interpretation,

                        'dcm_rows': rows, 'dcm_columns': columns}, index=[0]))

    return df
df_train = extract_DICOM_attributes('train')

train_dicom_df = train_df.merge(df_train, on='image_name')
train_dicom_df.head()
train_dicom_df.to_csv('train_dicom_df.csv', header=True, index=False) # todo
# todo
#train_img_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'

#test_img_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'
#from tqdm import tqdm

#from tqdm.keras import TqdmCallback

#from keras.preprocessing import image

#

#for data, location in zip([train_df, test_df],[train_img_path, test_img_path]):

#    images = data['image_name'].values

#    reds = np.zeros(images.shape[0])

#    greens = np.zeros(images.shape[0])

#    blues = np.zeros(images.shape[0])

#    mean = np.zeros(images.shape[0])

#    x = np.zeros(images.shape[0], dtype=int)

#    y = np.zeros(images.shape[0], dtype=int)

#    for i, path in enumerate(tqdm(images)):

#        img = np.array(image.load_img(os.path.join(location, f'{path}.jpg')))

#

#        reds[i] = np.mean(img[:,:,0].ravel())

#        greens[i] = np.mean(img[:,:,1].ravel())

#        blues[i] = np.mean(img[:,:,2].ravel())

#        mean[i] = np.mean(img)

#        x[i] = img.shape[1]

#        y[i] = img.shape[0]

#

#    data['reds'] = reds

#    data['greens'] = greens

#    data['blues'] = blues

#    data['mean_colors'] = mean

#    data['width'] = x

#    data['height'] = y

#

#train_df['total_pixels']= train_df['width']*train_df['height']

#test_df['total_pixels']= test_df['width']*test_df['height']

#train_df['res'] = train_df['width'].astype(str) + 'x' + train_df['height'].astype(str)

#test_df['res'] = test_df['width'].astype(str) + 'x' + test_df['height'].astype(str)
#train_df.head()
# Save the files

#train_df.to_csv('train_atr.csv', index=False)

#test_df.to_csv('test_atr.csv', index=False)
PATH_IMAGES = "/kaggle/input/imagesatr"



train_df = pd.read_csv(os.path.join(PATH_IMAGES, 'train_atr.csv'))

test_df = pd.read_csv(os.path.join(PATH_IMAGES, 'test_atr.csv'))
# todo
# Loading sample submission data

PATH = "/kaggle/input/siim-isic-melanoma-classification/"



sample_submission_df = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
# Getting dummy variables for gender on train set

sex_dummies = pd.get_dummies(train_df.sex, prefix='sex')

train_df = pd.concat([train_df, sex_dummies], axis=1)



# Now, on test set

sex_dummies = pd.get_dummies(test_df.sex, prefix='sex')

test_df = pd.concat([test_df, sex_dummies], axis=1)
train_df.head()
# Getting dummy variables for anatomy on train set

anatomy_dummies = pd.get_dummies(train_df.anatomy, prefix='anatomy')

train_df = pd.concat([train_df, anatomy_dummies], axis=1)



# Now, on test set

anatomy_dummies = pd.get_dummies(test_df.anatomy, prefix='anatomy')

test_df = pd.concat([test_df, anatomy_dummies], axis=1)
train_df.columns
# Removing white space 

train_df.columns = train_df.columns.str.replace(' ', '_')

train_df.columns = train_df.columns.str.replace('/', '_')



test_df.columns = test_df.columns.str.replace(' ', '_')

test_df.columns = test_df.columns.str.replace('/', '_')



# Dropping not usefull columns

train_df.drop(['image_name', 'ID','sex', 'anatomy', 'diagnosis', 'benign_malignant', 'res'], axis=1, inplace=True)

test_df.drop(['image_name', 'ID','sex', 'anatomy', 'res'], axis=1, inplace=True)
import xgboost as xgb



from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, cross_validate

from sklearn.metrics import roc_auc_score



X = train_df.drop(['target'], axis=1)

y = train_df.target
# Spliting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Fit model on all training data

model = xgb.XGBClassifier()

model.fit(X_train, y_train)



validation = model.predict_proba(X_test)[:, 1]



roc_auc_score(y_test, validation)
from sklearn import metrics

# plotando a curva ROC

y_pred_proba = model.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')

plt.legend(loc=4)
from xgboost import plot_importance

model = xgb.XGBClassifier()

model.fit(X, y)

# plot feature importance

plot_importance(model)

plt.show()
# Prediction 1

predictions = model.predict_proba(test_df)

metadata_df = pd.DataFrame(columns=['image_name', 'target'])



metadata_df['image_name'] = sample_submission_df['image_name']

metadata_df['target'] = predictions
# Making the prediction

model.fit(X_train, y_train)

predictions = model.predict_proba(test_df)[:, 1]



metadata_df = pd.DataFrame(columns=['image_name', 'target'])

metadata_df['image_name'] = sample_submission_df['image_name']

metadata_df['target'] = predictions
metadata_df.head()
# creating submission csv file

metadata_df.to_csv('submission_tabular.csv', header=True, index=False)
!pip install -q efficientnet
import os

import re



import numpy as np

import pandas as pd

import math



from matplotlib import pyplot as plt



from sklearn import metrics

from sklearn.model_selection import train_test_split



import tensorflow as tf

import tensorflow.keras.layers as L



import efficientnet.tfkeras as efn



from kaggle_datasets import KaggleDatasets
# Setting TPU as main device for training

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
# For tf.dataset

AUTO = tf.data.experimental.AUTOTUNE



# Data access

GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')



# Configuration

DEBUG = False

N_FOLD = 4

EPOCHS = 1 if DEBUG else 7

BATCH_SIZE = 8 * strategy.num_replicas_in_sync

IMAGE_SIZE = [1024, 1024]
sub = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')

test_files = tf.io.gfile.glob(GCS_PATH + '/tfrecords/test*.tfrec')
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.reshape(image, [*IMAGE_SIZE, 3])

    return image



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "image_name": tf.io.FixedLenFeature([], tf.string),

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    idnum = example['image_name']

    return image, idnum



def load_dataset(filenames, labeled=True, ordered=False):

    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    dataset = dataset.with_options(ignore_order)

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    return dataset



def get_test_dataset(test_files, ordered=False):

    dataset = load_dataset(test_files, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO)

    return dataset



def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)
def get_model():

    

    with strategy.scope():

        model = tf.keras.Sequential([

            efn.EfficientNetB3(

                input_shape=(*IMAGE_SIZE, 3),

                weights=None,

                include_top=False

            ),

            L.GlobalAveragePooling2D(),

            L.Dense(1, activation='sigmoid')

        ])

    

    return model
# Inference

from tqdm import tqdm



pred_df = pd.DataFrame()



tk0 = tqdm(range(N_FOLD), total=N_FOLD)



for fold in tk0:

    num_test = count_data_items(test_files)

    test_ds = get_test_dataset(test_files, ordered=True)

    test_images_ds = test_ds.map(lambda image, idnum: image)

    model = get_model()

    probabilities = model.predict(test_images_ds)

    test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

    test_ids = next(iter(test_ids_ds.batch(num_test))).numpy().astype('U')

    _pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities)})

    pred_df = pd.concat([pred_df, _pred_df])
mean_pred_df = pred_df.groupby('image_name', as_index=False).mean()

mean_pred_df.columns = ['image_name', 'target']
mean_pred_df.head()
pred_df.head()
# creating submission csv file

mean_pred_df.to_csv('submission_image.csv', header=True, index=False)
import pandas as pd

import matplotlib.pyplot as plt
image_sub = pd.read_csv('/kaggle/working/submission_image.csv')

tabular_sub = pd.read_csv('/kaggle/working/submission_tabular.csv')

tabular_sub.head()
submission = image_sub.copy()

submission.target = 0.9 * image_sub.target.values + 0.1 * tabular_sub.target.values

submission.to_csv('submission.csv',index=False)