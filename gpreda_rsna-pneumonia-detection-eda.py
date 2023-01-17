from datetime import datetime

dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

print(f"Updated {dt_string} (GMT)")
import pandas as pd 

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

from matplotlib.patches import Rectangle

import seaborn as sns

import pydicom as dcm

%matplotlib inline 

IS_LOCAL = False

import os

if(IS_LOCAL):

    PATH="../input/rsna-pneumonia-detection-challenge"

else:

    PATH="../input/"

print(os.listdir(PATH))
class_info_df = pd.read_csv(PATH+'/stage_2_detailed_class_info.csv')

train_labels_df = pd.read_csv(PATH+'/stage_2_train_labels.csv')                         
print(f"Detailed class info -  rows: {class_info_df.shape[0]}, columns: {class_info_df.shape[1]}")

print(f"Train labels -  rows: {train_labels_df.shape[0]}, columns: {train_labels_df.shape[1]}")
class_info_df.sample(10)
train_labels_df.sample(10)
def missing_data(data):

    total = data.isnull().sum().sort_values(ascending = False)

    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    return np.transpose(pd.concat([total, percent], axis=1, keys=['Total', 'Percent']))

missing_data(train_labels_df)
missing_data(class_info_df)
f, ax = plt.subplots(1,1, figsize=(6,4))

total = float(len(class_info_df))

sns.countplot(class_info_df['class'],order = class_info_df['class'].value_counts().index, palette='Set3')

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(100*height/total),

            ha="center") 

plt.show()
def get_feature_distribution(data, feature):

    # Get the count for each label

    label_counts = data[feature].value_counts()



    # Get total number of samples

    total_samples = len(data)



    # Count the number of items in each class

    print("Feature: {}".format(feature))

    for i in range(len(label_counts)):

        label = label_counts.index[i]

        count = label_counts.values[i]

        percent = int((count / total_samples) * 10000) / 100

        print("{:<30s}:   {} or {}%".format(label, count, percent))



get_feature_distribution(class_info_df, 'class')
train_class_df = train_labels_df.merge(class_info_df, left_on='patientId', right_on='patientId', how='inner')
train_class_df.sample(5)
fig, ax = plt.subplots(nrows=1,figsize=(12,6))

tmp = train_class_df.groupby('Target')['class'].value_counts()

df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index()

sns.barplot(ax=ax,x = 'Target', y='Exams',hue='class',data=df, palette='Set3')

plt.title("Chest exams class and Target")

plt.show()
target1 = train_class_df[train_class_df['Target']==1]

sns.set_style('whitegrid')

plt.figure()

fig, ax = plt.subplots(2,2,figsize=(12,12))

sns.distplot(target1['x'],kde=True,bins=50, color="red", ax=ax[0,0])

sns.distplot(target1['y'],kde=True,bins=50, color="blue", ax=ax[0,1])

sns.distplot(target1['width'],kde=True,bins=50, color="green", ax=ax[1,0])

sns.distplot(target1['height'],kde=True,bins=50, color="magenta", ax=ax[1,1])

locs, labels = plt.xticks()

plt.tick_params(axis='both', which='major', labelsize=12)

plt.show()
fig, ax = plt.subplots(1,1,figsize=(7,7))

target_sample = target1.sample(2000)

target_sample['xc'] = target_sample['x'] + target_sample['width'] / 2

target_sample['yc'] = target_sample['y'] + target_sample['height'] / 2

plt.title("Centers of Lung Opacity rectangles (brown) over rectangles (yellow)\nSample size: 2000")

target_sample.plot.scatter(x='xc', y='yc', xlim=(0,1024), ylim=(0,1024), ax=ax, alpha=0.8, marker=".", color="brown")

for i, crt_sample in target_sample.iterrows():

    ax.add_patch(Rectangle(xy=(crt_sample['x'], crt_sample['y']),

                width=crt_sample['width'],height=crt_sample['height'],alpha=3.5e-3, color="yellow"))

plt.show()
image_sample_path = os.listdir(PATH+'/stage_2_train_images')[:5]

print(image_sample_path)
image_train_path = os.listdir(PATH+'/stage_2_train_images')

image_test_path = os.listdir(PATH+'/stage_2_test_images')

print("Number of images in train set:", len(image_train_path),"\nNumber of images in test set:", len(image_test_path))
print("Unique patientId in  train_class_df: ", train_class_df['patientId'].nunique())      
tmp = train_class_df.groupby(['patientId','Target', 'class'])['patientId'].count()

df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index()

tmp = df.groupby(['Exams','Target','class']).count()

df2 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()

df2.columns = ['Exams', 'Target','Class', 'Entries']

df2
fig, ax = plt.subplots(nrows=1,figsize=(12,6))

sns.barplot(ax=ax,x = 'Target', y='Entries', hue='Exams',data=df2, palette='Set2')

plt.title("Chest exams class and Target")

plt.show()
samplePatientID = list(train_class_df[:3].T.to_dict().values())[0]['patientId']

samplePatientID = samplePatientID+'.dcm'

dicom_file_path = os.path.join(PATH,"stage_2_train_images/",samplePatientID)

dicom_file_dataset = dcm.read_file(dicom_file_path)

dicom_file_dataset
def show_dicom_images(data):

    img_data = list(data.T.to_dict().values())

    f, ax = plt.subplots(3,3, figsize=(16,18))

    for i,data_row in enumerate(img_data):

        patientImage = data_row['patientId']+'.dcm'

        imagePath = os.path.join(PATH,"stage_2_train_images/",patientImage)

        data_row_img_data = dcm.read_file(imagePath)

        modality = data_row_img_data.Modality

        age = data_row_img_data.PatientAge

        sex = data_row_img_data.PatientSex

        data_row_img = dcm.dcmread(imagePath)

        ax[i//3, i%3].imshow(data_row_img.pixel_array, cmap=plt.cm.bone) 

        ax[i//3, i%3].axis('off')

        ax[i//3, i%3].set_title('ID: {}\nModality: {} Age: {} Sex: {} Target: {}\nClass: {}\nWindow: {}:{}:{}:{}'.format(

                data_row['patientId'],

                modality, age, sex, data_row['Target'], data_row['class'], 

                data_row['x'],data_row['y'],data_row['width'],data_row['height']))

    plt.show()
show_dicom_images(train_class_df[train_class_df['Target']==1].sample(9))
def show_dicom_images_with_boxes(data):

    img_data = list(data.T.to_dict().values())

    f, ax = plt.subplots(3,3, figsize=(16,18))

    for i,data_row in enumerate(img_data):

        patientImage = data_row['patientId']+'.dcm'

        imagePath = os.path.join(PATH,"stage_2_train_images/",patientImage)

        data_row_img_data = dcm.read_file(imagePath)

        modality = data_row_img_data.Modality

        age = data_row_img_data.PatientAge

        sex = data_row_img_data.PatientSex

        data_row_img = dcm.dcmread(imagePath)

        ax[i//3, i%3].imshow(data_row_img.pixel_array, cmap=plt.cm.bone) 

        ax[i//3, i%3].axis('off')

        ax[i//3, i%3].set_title('ID: {}\nModality: {} Age: {} Sex: {} Target: {}\nClass: {}'.format(

                data_row['patientId'],modality, age, sex, data_row['Target'], data_row['class']))

        rows = train_class_df[train_class_df['patientId']==data_row['patientId']]

        box_data = list(rows.T.to_dict().values())

        for j, row in enumerate(box_data):

            ax[i//3, i%3].add_patch(Rectangle(xy=(row['x'], row['y']),

                        width=row['width'],height=row['height'], 

                        color="yellow",alpha = 0.1))   

    plt.show()
show_dicom_images_with_boxes(train_class_df[train_class_df['Target']==1].sample(9))
show_dicom_images(train_class_df[train_class_df['Target']==0].sample(9))
vars = ['Modality', 'PatientAge', 'PatientSex', 'BodyPartExamined', 'ViewPosition', 'ConversionType', 'Rows', 'Columns', 'PixelSpacing']



def process_dicom_data(data_df, data_path):

    for var in vars:

        data_df[var] = None

    image_names = os.listdir(PATH+data_path)

    for i, img_name in tqdm_notebook(enumerate(image_names)):

        imagePath = os.path.join(PATH,data_path,img_name)

        data_row_img_data = dcm.read_file(imagePath)

        idx = (data_df['patientId']==data_row_img_data.PatientID)

        data_df.loc[idx,'Modality'] = data_row_img_data.Modality

        data_df.loc[idx,'PatientAge'] = pd.to_numeric(data_row_img_data.PatientAge)

        data_df.loc[idx,'PatientSex'] = data_row_img_data.PatientSex

        data_df.loc[idx,'BodyPartExamined'] = data_row_img_data.BodyPartExamined

        data_df.loc[idx,'ViewPosition'] = data_row_img_data.ViewPosition

        data_df.loc[idx,'ConversionType'] = data_row_img_data.ConversionType

        data_df.loc[idx,'Rows'] = data_row_img_data.Rows

        data_df.loc[idx,'Columns'] = data_row_img_data.Columns  

        data_df.loc[idx,'PixelSpacing'] = str.format("{:4.3f}",data_row_img_data.PixelSpacing[0]) 
process_dicom_data(train_class_df,'stage_2_train_images/')
test_class_df = pd.read_csv(PATH+'/stage_2_sample_submission.csv')
test_class_df = test_class_df.drop('PredictionString',1)

process_dicom_data(test_class_df,'stage_2_test_images/')
print("Modalities: train:",train_class_df['Modality'].unique(), "test:", test_class_df['Modality'].unique())
print("Body Part Examined: train:",train_class_df['BodyPartExamined'].unique(), "test:", test_class_df['BodyPartExamined'].unique())
print("View Position: train:",train_class_df['ViewPosition'].unique(), "test:", test_class_df['ViewPosition'].unique())
get_feature_distribution(train_class_df,'ViewPosition')
def plot_window(data,color_point, color_window,text):

    fig, ax = plt.subplots(1,1,figsize=(7,7))

    plt.title("Centers of Lung Opacity rectangles over rectangles\n{}".format(text))

    data.plot.scatter(x='xc', y='yc', xlim=(0,1024), ylim=(0,1024), ax=ax, alpha=0.8, marker=".", color=color_point)

    for i, crt_sample in data.iterrows():

        ax.add_patch(Rectangle(xy=(crt_sample['x'], crt_sample['y']),

            width=crt_sample['width'],height=crt_sample['height'],alpha=3.5e-3, color=color_window))

    plt.show()
target1 = train_class_df[train_class_df['Target']==1]



target_sample = target1.sample(2000)

target_sample['xc'] = target_sample['x'] + target_sample['width'] / 2

target_sample['yc'] = target_sample['y'] + target_sample['height'] / 2



target_ap = target_sample[target_sample['ViewPosition']=='AP']

target_pa = target_sample[target_sample['ViewPosition']=='PA']
plot_window(target_ap,'green', 'yellow', 'Patient View Position: AP')
plot_window(target_pa,'blue', 'red', 'Patient View Position: PA')
get_feature_distribution(test_class_df,'ViewPosition')
print("Conversion Type: train:",train_class_df['ConversionType'].unique(), "test:", test_class_df['ConversionType'].unique())
print("Rows: train:",train_class_df['Rows'].unique(), "test:", test_class_df['Rows'].unique())

print("Columns: train:",train_class_df['Columns'].unique(), "test:", test_class_df['Columns'].unique())
tmp = train_class_df.groupby(['Target', 'PatientAge'])['patientId'].count()

df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index()

tmp = df.groupby(['Exams','Target', 'PatientAge']).count()

df2 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()
tmp = train_class_df.groupby(['class', 'PatientAge'])['patientId'].count()

df1 = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index()

tmp = df1.groupby(['Exams','class', 'PatientAge']).count()

df3 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()
fig, (ax) = plt.subplots(nrows=1,figsize=(16,6))

sns.barplot(ax=ax, x = 'PatientAge', y='Exams', hue='Target',data=df2)

plt.title("Train set: Chest exams Age and Target")

plt.xticks(rotation=90)

plt.show()
fig, (ax) = plt.subplots(nrows=1,figsize=(16,6))

sns.barplot(ax=ax, x = 'PatientAge', y='Exams', hue='class',data=df3)

plt.title("Train set: Chest exams Age and class")

plt.xticks(rotation=90)

plt.show()
target_age1 = target_sample[target_sample['PatientAge'] < 20]

target_age2 = target_sample[(target_sample['PatientAge'] >=20) & (target_sample['PatientAge'] < 35)]

target_age3 = target_sample[(target_sample['PatientAge'] >=35) & (target_sample['PatientAge'] < 50)]

target_age4 = target_sample[(target_sample['PatientAge'] >=50) & (target_sample['PatientAge'] < 65)]

target_age5 = target_sample[target_sample['PatientAge'] >= 65]
plot_window(target_age1,'blue', 'red', 'Patient Age: 1-19 years')
plot_window(target_age2,'blue', 'red', 'Patient Age: 20-34 years')
plot_window(target_age3,'blue', 'red', 'Patient Age: 35-49 years')
plot_window(target_age4,'blue', 'red', 'Patient Age: 50-65 years')
plot_window(target_age5,'blue', 'red', 'Patient Age: 65+ years')
fig, (ax) = plt.subplots(nrows=1,figsize=(16,6))

sns.countplot(test_class_df['PatientAge'], ax=ax)

plt.title("Test set: Patient Age")

plt.xticks(rotation=90)

plt.show()
tmp = train_class_df.groupby(['Target', 'PatientSex'])['patientId'].count()

df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index()

tmp = df.groupby(['Exams','Target', 'PatientSex']).count()

df2 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()

fig, ax = plt.subplots(nrows=1,figsize=(6,6))

sns.barplot(ax=ax, x = 'PatientSex', y='Exams', hue='Target',data=df2)

plt.title("Train set: Patient Sex and Target")

plt.show()
tmp = train_class_df.groupby(['class', 'PatientSex'])['patientId'].count()

df1 = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index()

tmp = df1.groupby(['Exams','class', 'PatientSex']).count()

df3 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()

fig, (ax) = plt.subplots(nrows=1,figsize=(6,6))

sns.barplot(ax=ax, x = 'PatientSex', y='Exams', hue='class',data=df3)

plt.title("Train set: Patient Sex and class")

plt.show()
target_female = target_sample[target_sample['PatientSex']=='F']

target_male = target_sample[target_sample['PatientSex']=='M']
plot_window(target_female,"red", "magenta","Patients Sex: Female")
plot_window(target_male,"darkblue", "blue", "Patients Sex: Male")
sns.countplot(test_class_df['PatientSex'])

plt.title("Test set: Patient Sex")

plt.show()