import numpy as np

import pandas as pd

import pydicom as pdm



import cv2

import matplotlib.pyplot as plt



import time

import os

from glob import glob

print(os.listdir("../input/"))
# path to data 

train_data_path = sorted(glob("../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/*.dcm"))

len(train_data_path)
# This is what the dcm file looks like

#idx = np.random.choice(len(train_data_path))

wtf = pdm.dcmread(train_data_path[191228])

# SOP Instance UID : http://www.otpedia.com/entryDetails.cfm?id=199

# Photometric Interpretation : https://www.dabsoft.ch/dicom/3/C.7.6.3.1.2/

wtf
sample_image = pdm.read_file(train_data_path[191228]).pixel_array

rgb = np.stack((sample_image,)*3, axis=-1)



fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,10))

ax[0].imshow(sample_image, "gray");   

ax[0].set_title('Original image', fontsize=20)

ax[1].imshow(rgb, cmap='binary');

ax[1].set_title('RGB Image',fontsize=20)

fig.subplots_adjust(hspace=0, wspace=0.1)

# Dataset with unique UID rows(image data in csv format), and original dataset.

all_data = pd.read_csv("../input/rsna-ihd/all_data_as_df.csv")

df = pd.read_csv("../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv")

df.head()
# Code from https://www.kaggle.com/akensert/inceptionv3-prev-resnet50-keras-baseline-model/notebook

df["SOP Instance UID"] = df["ID"].str.slice(stop=12)

df["Diagnosis"] = df["ID"].str.slice(start=13)

df = df.loc[:, ["Label", "Diagnosis", "SOP Instance UID"]]



df[10:20]
df_without_d = df[(df["Diagnosis"] == "any") & (df["Label"] == 0)]

df_without_d = df_without_d.drop_duplicates(subset=['SOP Instance UID'])



# Merge 2 df

df_without_d = pd.merge(df_without_d, all_data, on='SOP Instance UID')



# For symmetry, we take only 1000 samples.

df_without_d = df_without_d.reset_index(drop=True)

indexes = np.random.randint(0, len(df_without_d), 1000)

df_without_d = df_without_d.loc[indexes]



df_without_d['values'] = 'Without any diagnosis'
# Label Coding

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(["intraparenchymal", "intraventricular", "epidural", "subarachnoid", "any", "subdural"])

df['Coded Label'] = le.fit_transform(df['Diagnosis'])



df[500:510]
# Extract all rows that do not contain “any” and which are tagged as 1

df = df[(df["Diagnosis"] != "any") & (df["Label"] == 1)]

df = df.reset_index(drop=True)



# Merge 2 df

df_with_diagnosis = pd.merge(df, all_data, on='SOP Instance UID')



print(df_with_diagnosis.shape)

df_with_diagnosis[200:205]
# Distribution of the number of unique samples



uid = len(df_with_diagnosis['SOP Instance UID'].value_counts())

patient_id  = len(df_with_diagnosis['Patient ID'].value_counts())

total_labels = len(df_with_diagnosis)



distr = [patient_id, uid, total_labels]



plt.figure(figsize=(15,7))

plt.title('Distribution of the number of unique samples',fontsize=15)

plt.bar(['Patient ID', 'SOP Instance UID', 'Labels'], distr,

            color=['purple','lime',"gold"]);

plt.ylabel('Number of samples',fontsize=15);

unique_uid = df_with_diagnosis.groupby(['SOP Instance UID']).size().reset_index(name='Counts')

unique_uid.head(10)
%%time

def make_group_df(x_df, num_label):

    main_df = df_with_diagnosis

    df = x_df[x_df['Counts'] == num_label]

    df = df.reset_index(drop=True)

    df = pd.merge(df, main_df, on='SOP Instance UID')

    return df



label1 = make_group_df(unique_uid, 1)

label2 = make_group_df(unique_uid, 2)

label3 = make_group_df(unique_uid, 3)

label4 = make_group_df(unique_uid, 4)

label5 = make_group_df(unique_uid, 5)
def make_df(x_df):

    awesom_dict = {}

    main_df = df_with_diagnosis

    for i in range(0, len(x_df), 3) :

        k = x_df[x_df['SOP Instance UID'] == x_df['SOP Instance UID'][i]]

        v = k['Diagnosis'].values#to_string()

        v = ' '.join(v)

        k = k['SOP Instance UID'].values[0]

        awesom_dict[k] = v

    df = pd.DataFrame(list(awesom_dict.items()), columns=['SOP Instance UID', 'values'])

    df = pd.merge(main_df, df, on='SOP Instance UID')  

    return df
%%time

label1 = make_df(label1)

label2 = make_df(label2)

label3 = make_df(label3)

label4 = make_df(label4)

label5 = make_df(label5)

label5.shape, label4.shape, label3.shape, label2.shape, label1.shape
# Images with 1 labels

print(label1['values'].value_counts())

label1_1 = label1[label1['values'] == 'subdural']

label1_2 = label1[label1['values'] == 'subarachnoid']

label1_3 = label1[label1['values'] == 'intraparenchymal']

label1_4 = label1[label1['values'] == 'intraventricular']

label1_5 = label1[label1['values'] == 'epidural']
# Images with 2 labels

value_counts = label2['values'].value_counts()

print(f'Number of unique label groups:{len(value_counts)}\n\n{value_counts}')



label2_1 = label2[label2['values'] == value_counts.index[0]]

label2_2 = label2[label2['values'] == value_counts.index[1]]

label2_3 = label2[label2['values'] == value_counts.index[2]]

label2_4 = label2[label2['values'] == value_counts.index[3]]

label2_5 = label2[label2['values'] == value_counts.index[4]]

label2_6 = label2[label2['values'] == value_counts.index[5]]

label2_7 = label2[label2['values'] == value_counts.index[6]]

label2_8 = label2[label2['values'] == value_counts.index[7]]

label2_9 = label2[label2['values'] == value_counts.index[8]]

label2_10 = label2[label2['values'] == value_counts.index[9]]
# Images with 3 labels

value_counts = label3['values'].value_counts()

print(f'Number of unique label groups:{len(value_counts)}\n\n{value_counts}')



label3_1 = label3[label3['values'] == value_counts.index[0]]

label3_2 = label3[label3['values'] == value_counts.index[1]]

label3_3 = label3[label3['values'] == value_counts.index[2]]

label3_4 = label3[label3['values'] == value_counts.index[3]]

label3_5 = label3[label3['values'] == value_counts.index[4]]

label3_6 = label3[label3['values'] == value_counts.index[5]]

label3_7 = label3[label3['values'] == value_counts.index[6]]

label3_8 = label3[label3['values'] == value_counts.index[7]]

label3_9 = label3[label3['values'] == value_counts.index[8]]

label3_10 = label3[label3['values'] == value_counts.index[9]]
# Images with 4 labels

value_counts = label4['values'].value_counts()

print(f'Number of unique label groups:{len(value_counts)}\n\n{value_counts}')



label4_1 = label4[label4['values'] == value_counts.index[0]]

label4_2 = label4[label4['values'] == value_counts.index[1]]

label4_3 = label4[label4['values'] == value_counts.index[2]]

label4_4 = label4[label4['values'] == value_counts.index[3]]

label4_5 = label4[label4['values'] == value_counts.index[4]]
# Images with 5 labels

value_counts = label5['values'].value_counts()

print(f'Number of unique label groups:{len(value_counts)}\n\n{value_counts}')



label5 = label5[label5['values'] == value_counts.index[0]]

label5 = label5.drop_duplicates(subset=['SOP Instance UID'])
def create_description(row):

    

   # Create description for video

   # row: row of dataframe with label values



    label = row.split(' ')

    img = np.ones((512,512, 3), dtype=np.uint8)

    

    # Description parameters

    if len(label) == 1:

        font_size = 1.5

        y0 = 200

        pad = 60

    elif len(label) > 2:

        font_size = 1.5

        y0 = 150

        pad = 60

    else:

        font_size = 2

        y0 = 200

        pad = 100

        

    for i, line in enumerate(label):

        y = y0 + i*pad

        cv2.putText(img, line.capitalize(),

                    (50, y ), cv2.FONT_ITALIC,

                    font_size,(0,255,127),

                    2,cv2.LINE_AA)

    return img
def data_generator(df):



    # Calculate batch size, &

    # list of indices to remove from df



    batch_size = len(df)// int(np.sqrt(len(df)))

    del_indices=[]



    while len(df) >= batch_size:

        df = df.drop(index=del_indices) 

        df = df.reset_index(drop=True)



        batch_i = []

        del_indices = []

  

        if len(df) != 0:

            for i in range(batch_size):

                image = df['path'].sample(1).values[0]

                # Read image as np.array (512x512)

                image = pdm.read_file(image).pixel_array

                

                # Color map and normalization instance

                cmap = plt.cm.bone

                norm = plt.Normalize(vmin=image.min(), vmax=image.max())

                

                # image is now RGBA (512x512x4) 

                image = cmap(norm(image))



                batch_i += [image]

                del_indices += [i]



            yield batch_i, del_indices
def make_video(x_df, fps=5):

    

    # Remove duplicates

    x_df = x_df.drop_duplicates(subset=['SOP Instance UID'])

    x_df = x_df.reset_index(drop=True)

    

    # Create description

    values = x_df['values'][0]

    description = create_description(values)

    description = [description] * 25

    video_name = '-'.join(x_df['values'][0].split()) + '.avi'

    

    # Define the codec and create VideoWrite object

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    video = cv2.VideoWriter(video_name, fourcc, fps, (512, 512))

    

    # Write the video

    for d in description:

        video.write(d.astype('uint8')) 

     

    for batch, _ in data_generator(x_df):    

        # Take only 3 channels and normalize

        # values to val [0.0, 1.0]

        for img in batch:

            img = img[:,:,:3] * 255 

            video.write(img.astype('uint8')) 
# Grouped Data List

grouped_dfs = [label1_1, label1_2, label1_3, label1_4, label1_5,

              label2_1, label2_2, label2_3, label2_4, label2_5,

              label2_6, label2_7, label2_8, label2_9, label2_10,

              label3_1, label3_2, label3_3, label3_4, label3_5,

              label3_6, label3_7, label3_8, label3_9, label3_10,

              label4_1, label4_2, label4_3, label4_4, label4_5,

              label5, df_without_d]
%%time

# Recursive function call

def record_all_videos(data_list):

    a = len(data_list)

    while a !=0:

        a -= 1

        time.sleep(5)

        print(f"{time.ctime()} creating video № {a}")

        make_video(data_list[a]) #df_img_video

record_all_videos(grouped_dfs)
# Our outputs

!ls
video_names = [f'''

subdural.avi

subarachnoid.avi

intraparenchymal.avi

intraventricular.avi

epidural.avi

intraparenchymal-intraventricular.avi

subarachnoid-subdural.avi

intraparenchymal-subarachnoid.avi

intraventricular-subarachnoid.avi

intraparenchymal-subdural.avi

intraventricular-subdural.avi

epidural-subdural.avi

epidural-intraparenchymal.avi

epidural-subarachnoid.avi

epidural-intraventricular.avi

intraparenchymal-subarachnoid-subdural.avi

intraparenchymal-intraventricular-subarachnoid.avi

intraventricular-subarachnoid-subdural.avi

intraparenchymal-intraventricular-subdural.avi

epidural-subarachnoid-subdural.avi

epidural-intraparenchymal-subdural.avi

epidural-intraparenchymal-subarachnoid.avi

epidural-intraparenchymal-intraventricular.avi

epidural-intraventricular-subdural.avi

epidural-intraventricular-subarachnoid.avi

intraparenchymal-intraventricular-subarachnoid-subdural.avi

epidural-intraparenchymal-subarachnoid-subdural.avi

epidural-intraparenchymal-intraventricular-subarachnoid.avi

epidural-intraventricular-subarachnoid-subdural.avi

epidural-intraparenchymal-intraventricular-subdural.avi

epidural-intraparenchymal-intraventricular-subarachnoid-subdural.avi

Without-any-diagnosis.avi

''']



with open('join.txt', 'a') as f:

    for i in video_names[0].split():

        f.write(f"file {i}\n")
!cat join.txt
# FFmpeg installation

!apt install -y ffmpeg > /dev/null
# Concatenate all videos

!ffmpeg -f concat -i join.txt -c copy output.avi
! wget -O audio.mp3 http://dl17.y2mate.com/?file=M3R4SUNiN3JsOHJ6WWQ2a3NQS1Y5ZGlxVlZIOCtyaDNwOTAxMGdJd1BxUkttWWtxd3ZlZkp0RndMNm9OeEkrc0JlRlo0QzNSY3UrR01RR0N2NVlpZlhlSTRkODV2RHZ3NElFMVY4MWxSQmY3a05tdW55bEpnUmJLYVlENU1vSnZiaVZqdDBOY2dIVER4L1BFcWx2MW95N3JvVVNQYVJzYXN5OERZTENDMTRaSHdIM2ViZktoZ2JZSXZYWGFzTHdlbjZLN3JWUDd3cVk3OXQ5NlV3ODVPTU5pMFpYZ3plUHpxa0VtaEp3SnlYS2hzZDJuQzV3OEdhdUtSanBsTFNzSzF1THVVaEFYd1NFaC9HV28vNnNnNmw4YWZyVjc3SGFtK1BIaVdUS1JadjMrWHE2VGN0eWFuUHo2c09GMXZINlM5TFBNbm9wVDJBVzNHNDc4U3RnTnYxc0trZjdVdklKbmt3YjNpMTFEekxKWWxnMzViMXcwWEpKZktub0tlcDFDVUg5Tis1NjN1UFk9
!ffmpeg -i output.avi -i audio.mp3 -c copy -map 0:v:0 -map 1:a:0 -shortest Result.avi
# Extract duration from all videos

def get_duration(video_list):

    duration_list = []

    for video in video_list:

        cap = cv2.VideoCapture(video)

        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        duration = frame_count/fps

        duration_list.append(duration)

        cap.release()

    return duration_list



d_list = get_duration(video_names[0].split())
# Recursive function call

def make_timecodes(video_list):

    a = len(video_list)

    timecodes = []

    while a !=0:

        a -= 1

        duration = sum(video_list[:a])

        minutes = int(duration/60)

        seconds = duration%60

        time = f"{minutes}:{np.around(seconds,2)}"

        timecodes.append(time)

    with open('timecodes.txt', 'a') as f:

        f.write("Time Codes in minutes:\n")

        for n, t in zip(video_names[0].split(),timecodes[::-1]):  

            f.write(f"{n} - {t}\n")



make_timecodes(d_list)
!cat timecodes.txt
! mkdir output

! cp Result.avi timecodes.txt output
from IPython.display import FileLink, FileLinks

# create a IPython.display.FileLink object with provided file name and path.

output_files = FileLinks(path='output')



# print the FileLinks objects.

output_files

from IPython.display import YouTubeVideo

# Getting video like this

YouTubeVideo('ZtF2Aq0d-J4')