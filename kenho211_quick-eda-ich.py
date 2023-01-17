import numpy as np

import pandas as pd

import os

import pydicom

import matplotlib.pyplot as plt

import seaborn as sns
TRAIN_DIR = '../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/'

TEST_DIR = '../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/'



train_img = [TRAIN_DIR + f for f in os.listdir(TRAIN_DIR)]

test_img = [TEST_DIR + f for f in os.listdir(TEST_DIR)]
os.listdir('../input/rsna-intracranial-hemorrhage-detection')
sample_submission = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv')

df_train = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')
# Try to use pydicom to analyze the images

filename = train_img[0]

ds = pydicom.dcmread(filename)

print(ds)

plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
# Clean the dataset

df_train['id_code'] = df_train.ID.apply(lambda x: 'ID_'+x.split('_')[-2])

df_train['subtype'] = df_train.ID.apply(lambda x: x.split('_')[-1])
# Check if sub-type are mutually exclusive

df_train[['id_code','Label']].groupby('id_code').sum().Label.value_counts()
subtype_lst = df_train.subtype.unique()



df = pd.DataFrame()

df['id_code'] = df_train.id_code.unique()

for n in subtype_lst:

    temp_df = df_train[df_train.subtype==n][['id_code','Label']].rename(columns={'Label': n})

    df = df.merge(temp_df, on='id_code', how='left')

    del temp_df
def get_first_of_dicom_field_as_int(x):

    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)

    if type(x) == pydicom.multival.MultiValue:

        return int(x[0])

    else:

        return int(x)



def get_windowing(data):

    dicom_fields = [data[('0028','1050')].value, #window center

                    data[('0028','1051')].value, #window width

                    data[('0028','1052')].value, #intercept

                    data[('0028','1053')].value] #slope

    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]



def dcm2image(data):

    window_center, window_width, intercept, slope = get_windowing(data)

    img = data.pixel_array

    img = (img*slope +intercept)

    img_min = window_center - window_width//2

    img_max = window_center + window_width//2

    img[img<img_min] = img_min

    img[img>img_max] = img_max

    return img 
# matplotlib.cm options: Sequential (2)

#['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 

#'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',

#'hot', 'afmhot', 'gist_heat', 'copper']
SUBTYPE = 'intraparenchymal'

fig=plt.figure(figsize=(30, 20))

fig.suptitle(SUBTYPE, fontsize=20)

columns = 5; rows = 4

#lst = df_train[(df_train.subtype==SUBTYPE)&(df_train.Label==1)].head(columns*rows).id_code.tolist()

lst = df[df[SUBTYPE]==1].id_code.tolist()



for i in range(1, columns*rows +1):

    ds = pydicom.dcmread(TRAIN_DIR + lst[i-1]+'.dcm')

    fig.add_subplot(rows, columns, i)

    

    img = dcm2image(ds)

    plt.imshow(img, cmap=plt.cm.bone)

    fig.add_subplot
SUBTYPE = 'intraventricular'

fig=plt.figure(figsize=(30, 20))

fig.suptitle(SUBTYPE, fontsize=20)

columns = 5; rows = 4

#lst = df_train[(df_train.subtype==SUBTYPE)&(df_train.Label==1)].head(columns*rows).id_code.tolist()

lst = df[df[SUBTYPE]==1].id_code.tolist()



for i in range(1, columns*rows +1):

    ds = pydicom.dcmread(TRAIN_DIR + lst[i-1]+'.dcm')

    fig.add_subplot(rows, columns, i)

    

    img = dcm2image(ds)

    plt.imshow(img, cmap=plt.cm.bone)

    fig.add_subplot
SUBTYPE = 'subarachnoid'

fig=plt.figure(figsize=(30, 20))

fig.suptitle(SUBTYPE, fontsize=20)

columns = 5; rows = 4

#lst = df_train[(df_train.subtype==SUBTYPE)&(df_train.Label==1)].head(columns*rows).id_code.tolist()

lst = df[df[SUBTYPE]==1].id_code.tolist()



for i in range(1, columns*rows +1):

    ds = pydicom.dcmread(TRAIN_DIR + lst[i-1]+'.dcm')

    fig.add_subplot(rows, columns, i)

    

    img = dcm2image(ds)

    plt.imshow(img, cmap=plt.cm.bone)

    fig.add_subplot
SUBTYPE = 'subdural'

fig=plt.figure(figsize=(30, 20))

fig.suptitle(SUBTYPE, fontsize=20)

columns = 5; rows = 4

#lst = df_train[(df_train.subtype==SUBTYPE)&(df_train.Label==1)].head(columns*rows).id_code.tolist()

lst = df[df[SUBTYPE]==1].id_code.tolist()



for i in range(1, columns*rows +1):

    ds = pydicom.dcmread(TRAIN_DIR + lst[i-1]+'.dcm')

    fig.add_subplot(rows, columns, i)

    

    img = dcm2image(ds)

    plt.imshow(img, cmap=plt.cm.bone)

    fig.add_subplot
SUBTYPE = 'epidural'

fig=plt.figure(figsize=(30, 20))

fig.suptitle(SUBTYPE, fontsize=20)

columns = 5; rows = 4

#lst = df_train[(df_train.subtype==SUBTYPE)&(df_train.Label==1)].head(columns*rows).id_code.tolist()

lst = df[df[SUBTYPE]==1].id_code.tolist()



for i in range(1, columns*rows +1):

    ds = pydicom.dcmread(TRAIN_DIR + lst[i-1]+'.dcm')

    fig.add_subplot(rows, columns, i)

    

    img = dcm2image(ds)

    plt.imshow(img, cmap=plt.cm.bone)

    fig.add_subplot
SUBTYPE = 'any'

fig=plt.figure(figsize=(30, 20))

fig.suptitle('HEALTHY', fontsize=20)

columns = 5; rows = 4

#lst = df_train[(df_train.subtype==SUBTYPE)&(df_train.Label==0)].head(columns*rows).id_code.tolist()

lst = df[df[SUBTYPE]==0].id_code.tolist()



for i in range(1, columns*rows +1):

    ds = pydicom.dcmread(TRAIN_DIR + lst[i-1]+'.dcm')

    fig.add_subplot(rows, columns, i)

    

    img = dcm2image(ds)

    plt.imshow(img, cmap=plt.cm.bone)

    fig.add_subplot
SUBTYPE = 'any'

fig=plt.figure(figsize=(30, 20))

fig.suptitle('SUPER UNHEALTHY', fontsize=20)

columns = 5; rows = 4



mask = (df_train[['id_code','Label']].groupby('id_code').sum()==6).values

#lst = df_train[['id_code','Label']].groupby('id_code').sum()[mask].reset_index().id_code.tolist()

lst = df[df.sum(1) == 6].id_code.tolist()



for i in range(1, columns*rows +1):

    ds = pydicom.dcmread(TRAIN_DIR + lst[i-1]+'.dcm')

    fig.add_subplot(rows, columns, i)

    

    img = dcm2image(ds)

    plt.imshow(img, cmap=plt.cm.bone)

    fig.add_subplot
SUBTYPE = 'any'

fig=plt.figure(figsize=(30, 20))

fig.suptitle('SAME PATIENT?', fontsize=20)

columns = 3; rows = 2



mask = (df_train[['id_code','Label']].groupby('id_code').sum()==6).values

lst = df[df.sum(1) == 6].id_code.tolist()

idx = [0, 1, 4, 7, 11,15]

select_lst = [lst[i] for i in idx]



for i in range(1, columns*rows +1):

    ds = pydicom.dcmread(TRAIN_DIR + select_lst[i-1]+'.dcm')

    fig.add_subplot(rows, columns, i)

    

    img = dcm2image(ds)

    plt.imshow(img, cmap=plt.cm.bone)

    fig.add_subplot
# Look into the details

for n in select_lst:

    filename = TRAIN_DIR + select_lst[0] + '.dcm'

    ds = pydicom.dcmread(filename)

    print(ds.PatientID)
# These images are from the same patient with different CT slice

# Maybe we should group these images by PatientID and split by Patient ID in train-val split



# P.S. I just learnt that having different slices of the same patient is a normal pratice of CT scans,

# so bear with me if the finding above looks stupid to you haha
pd.DataFrame(df[subtype_lst[0]].value_counts()).rename(columns={'epidural':'n'})
dist = pd.DataFrame()

for n in subtype_lst:

    dist[n+'_count'] = df[n].value_counts().values

    dist[n+'_perc'] = df[n].value_counts(normalize=True).values
dist
fig=plt.figure(figsize=(24, 5))

fig.suptitle('Distribution of sub-types', fontsize=10)

columns = 6; rows = 1



lst = df.columns.tolist()

lst.remove('id_code')

for i in range(1, columns*rows +1):

    ax = fig.add_subplot(rows, columns, i)

    ax.set_title(lst[i-1], fontsize=10)

    #ax.bar([0,1], dist[lst[i-1]+'_perc'], color=['xkcd:sky blue', 'xkcd:orange'])

    sns.barplot([0,1], dist[lst[i-1]+'_perc'])

    fig.add_subplot
sns.heatmap(df.corr(method='pearson'))
#ds.PatientID

#ds.StudyInstanceUID

#ds.SeriesInstanceUID
patient_lst = []

study_lst = []

series_lst = []

for n in df.id_code.values:

    temp_ds = pydicom.dcmread(TRAIN_DIR + n + '.dcm')

    patient_lst.append(temp_ds.PatientID)

    study_lst.append(temp_ds.StudyInstanceUID)

    series_lst.append(temp_ds.SeriesInstanceUID)



df['PatientID'] = patient_lst

df['StudyInstanceUID'] = study_lst

df['SeriesInstanceUID'] = series_lst
df.to_csv('train_df_with_UID.csv')