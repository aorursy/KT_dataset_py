import pandas as pd 

import numpy as np

import os

import matplotlib

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

from matplotlib.patches import Rectangle

import pydicom as dcm

import seaborn as sns

%matplotlib inline 

PATH = "/kaggle/input/siim-isic-melanoma-classification/"
sample_submission_df = pd.read_csv(os.path.join(PATH,'sample_submission.csv'))

train_df = pd.read_csv(os.path.join(PATH,'train.csv'))

test_df = pd.read_csv(os.path.join(PATH,'test.csv'))
print(f"sample submission shape: {sample_submission_df.shape}")

print(f"train shape: {train_df.shape}")

print(f"test shape: {test_df.shape}")
sample_submission_df.head()
train_df.head()
test_df.head()
train_image_list = os.listdir(os.path.join(PATH, 'train'))

test_image_list = os.listdir(os.path.join(PATH, 'test'))



print(f"train image_name list: {train_df.image_name.nunique()}")

print(f"train image list: {len(train_image_list)}")

print(f"test image_name list: {test_df.image_name.nunique()}")

print(f"test image list: {len(test_image_list)}")
trimmed_train_image_list = []

for img in train_image_list:

    trimmed_train_image_list.append(img.split('.dcm')[0])

    

trimmed_test_image_list = []

for img in test_image_list:

    trimmed_test_image_list.append(img.split('.dcm')[0])  
intersect_train_train = (set(train_df.image_name.unique()) & set(trimmed_train_image_list))

intersect_test_test = (set(test_df.image_name.unique()) & set(trimmed_test_image_list))



print(f"image train (dcm) & train csv: {len(intersect_train_train)}")

print(f"image test (dcm) & test csv: {len(intersect_test_test)}")
tr_patient_id = train_df.patient_id.nunique()

te_patient_id = test_df.patient_id.nunique()

list_tr_patient_id = train_df.patient_id.unique()

list_te_patient_id = test_df.patient_id.unique()

intersection = set(list_tr_patient_id) & set(list_te_patient_id)

print(f"Unique patients in train: {tr_patient_id} and test: {te_patient_id}")

print(f"Patients in common in train and test: {len(intersection)}")
tmp = train_df.groupby(['patient_id', 'target'])['image_name'].count()

tr_df = pd.DataFrame(tmp).reset_index(); tr_df.columns = ['patient_id', 'target', 'images']



tmp = test_df.groupby(['patient_id'])['image_name'].count()

te_df = pd.DataFrame(tmp).reset_index(); te_df.columns = ['patient_id', 'images']





tr_df.head()
te_df.head()
plt.figure()

fig, ax = plt.subplots(1,3,figsize=(16,6))

g_tr0 = sns.distplot(tr_df.loc[tr_df.target==0, 'images'],kde=False,bins=50, color="green",label='target = 0', ax=ax[0])

g_tr1 = sns.distplot(tr_df.loc[tr_df.target==1, 'images'],kde=False,bins=50, color="red",label='target = 1', ax=ax[1])

g_te = sns.distplot(te_df['images'],kde=False,bins=50, color="blue", label='columns', ax=ax[2])

g_tr0.set_title('Number of images / patient - Train set\n target = 0 (benign)')

g_tr1.set_title('Number of images / patient - Train set\n target = 1 (malignant)')

g_te.set_title('Number of images / patient - Test set')



locs, labels = plt.xticks()

plt.tick_params(axis='both', which='major', labelsize=12)

plt.show()
def plot_count(df, feature, title='', size=2, rotate_axis = False):

    f, ax = plt.subplots(1,1, figsize=(3*size,2*size))

    total = float(len(df))

    sns.countplot(df[feature],order = df[feature].value_counts().index, palette='Set3')

    plt.title(title)

    if(rotate_axis):

        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
plot_count(train_df, 'sex', 'Patient sex (train) - data count and percent')
plot_count(test_df, 'sex', 'Patient sex (test) - data count and percent')
plot_count(test_df, 'age_approx', 'Patient approximate age (train) - data count and percent', size=4)
plot_count(test_df, 'age_approx', 'Patient approximate age (test) - data count and percent', size=4)
plot_count(train_df, 'anatom_site_general_challenge', 'Location of imaged site/anatomy part (train) - data count and percent', size=3)
plot_count(test_df, 'anatom_site_general_challenge', 'Location of imaged site/anatomy part (test) - data count and percent', size=3)
plot_count(train_df, 'diagnosis', 'Detailed diagnosis (train) - data count and percent', size=4, rotate_axis=True)
plot_count(train_df, 'benign_malignant', 'Indicator of malignancy of imaged lesion  (train) - data count and percent')
plot_count(train_df, 'target', 'Target value - 0: bening, 1: malignant (train)\n data count and percent')
fig, ax = plt.subplots(nrows=1,figsize=(16,6)) 

tmp = train_df.groupby('diagnosis')['target'].value_counts() 

df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index() 

sns.barplot(ax=ax,x = 'diagnosis', y='Exams',hue='target',data=df, palette='Set1') 

plt.title("Number of examinations grouped on Diagnosis and Target") 

plt.show()
fig, ax = plt.subplots(nrows=1,figsize=(16,6)) 

tmp = train_df.groupby('diagnosis')['benign_malignant'].value_counts() 

df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index() 

sns.barplot(ax=ax,x = 'diagnosis', y='Exams',hue='benign_malignant',data=df, palette='Set1') 

plt.title("Number of examinations grouped on Diagnosis and Benign/Malignant") 

plt.show()
fig, ax = plt.subplots(nrows=1,figsize=(16,6)) 

tmp = train_df.groupby('diagnosis')['sex'].value_counts() 

df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index() 

sns.barplot(ax=ax,x = 'diagnosis', y='Exams',hue='sex',data=df, palette='Set1') 

plt.title("Number of examinations grouped on Diagnosis and Sex") 

plt.show()
fig, ax = plt.subplots(nrows=1,figsize=(8,6)) 

tmp = train_df.groupby('sex')['target'].value_counts() 

df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index() 

sns.barplot(ax=ax,x = 'sex', y='Exams',hue='target',data=df, palette='Set1') 

plt.title("Number of examinations grouped on Sex and Target") 

plt.show()
def plot_distribution_grouped(feature, feature_group, hist_flag=True):

    fig, ax = plt.subplots(nrows=1,figsize=(12,6)) 

    for f in train_df[feature_group].unique():

        df = train_df.loc[train_df[feature_group] == f]

        sns.distplot(df[feature], hist=hist_flag, label=f)

    plt.title(f'Data/image {feature} distribution, grouped by {feature_group}')

    plt.legend()

    plt.show()
plot_distribution_grouped('age_approx', 'sex')
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

        ax[i//3, i%3].set_title(f"ID: {data_row['image_name']}\nModality: {modality} Age: {age} Sex: {sex}\nDiagnosis: {data_row['diagnosis']}")

    plt.show()
show_dicom_images(train_df[train_df['target']==1].sample(9))
show_dicom_images(train_df[train_df['target']==0].sample(9))
show_dicom_images(train_df[train_df['diagnosis']=='melanoma'].sample(9))
show_dicom_images(train_df[train_df['anatom_site_general_challenge']=='head/neck'].sample(9))
show_dicom_images(train_df[train_df['anatom_site_general_challenge']=='torso'].sample(9))
sample_item = list(train_df[:3].T.to_dict().values())[0]

sample_image_name = sample_item['image_name']

sample_patient_sex = sample_item['sex']

sample_patient_age = sample_item['age_approx']

body_part_examined = sample_item['anatom_site_general_challenge']

image_ID = sample_image_name +'.dcm'

dicom_file_path = os.path.join(PATH,"train/",image_ID)

dicom_file_dataset = dcm.read_file(dicom_file_path)

print(sample_image_name, sample_patient_sex, sample_patient_age,body_part_examined)

dicom_file_dataset
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
tr_df = extract_DICOM_attributes('train')

train_dicom_df = train_df.merge(tr_df, on='image_name')
train_dicom_df.head()
te_df = extract_DICOM_attributes('test')

test_dicom_df = test_df.merge(te_df, on='image_name')
test_dicom_df.head()
plt.figure()

fig, ax = plt.subplots(1,2,figsize=(12,6))

g0 = sns.distplot(train_dicom_df['dcm_rows'],kde=False,bins=50, color="green",label='rows', ax=ax[0])

g1 = sns.distplot(train_dicom_df['dcm_columns'],kde=False,bins=50, color="magenta", label='columns', ax=ax[1])

g0.set_title('Train set - rows distribution')

g1.set_title('Train set - columns distribution')

locs, labels = plt.xticks()

plt.tick_params(axis='both', which='major', labelsize=12)

plt.show()
plt.figure()

fig, ax = plt.subplots(1,2,figsize=(12,6))

g0 = sns.distplot(test_dicom_df['dcm_rows'],kde=False,bins=50, color="green",label='rows', ax=ax[0])

g1 = sns.distplot(test_dicom_df['dcm_columns'],kde=False,bins=50, color="magenta", label='columns', ax=ax[1])

g0.set_title('Test set - rows distribution')

g1.set_title('Test set - columns distribution')

locs, labels = plt.xticks()

plt.tick_params(axis='both', which='major', labelsize=12)

plt.show()