import pydicom
import os
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
import cv2
import seaborn as sns
from tqdm import tqdm
def show_dcm_info(dataset):
    print("Filename.........:", file_path)
    print("Storage type.....:", dataset.SOPClassUID)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name......:", display_name)
    print("Patient id..........:", dataset.PatientID)
    print("Patient's Age.......:", dataset.PatientAge)
    print("Patient's Sex.......:", dataset.PatientSex)
    print("Modality............:", dataset.Modality)
    print("Body Part Examined..:", dataset.BodyPartExamined)
    print("View Position.......:", dataset.ViewPosition)
    
    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)
def plot_pixel_array(dataset, figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()
i = 1
num_to_plot = 5
for file_name in os.listdir('../input/stage_1_train_images/'):
    file_path = os.path.join('../input/stage_1_train_images/', file_name)
    dataset = pydicom.dcmread(file_path)
    show_dcm_info(dataset)
    plot_pixel_array(dataset)
    
    if i >= num_to_plot:
        break
    
    i += 1
train_demo_df = pd.DataFrame()
ids = []
ages = []
sexs = []
img_avg_lums = []
img_max_lums = []
img_min_lums = []

from multiprocessing.pool import Pool, ThreadPool

pool = ThreadPool(4)

def process_image(dataset):
    _id = dataset.PatientID
    _age = dataset.PatientAge
    _sex = dataset.PatientSex
    _mean = np.mean(dataset.pixel_array)
    _min = np.max(dataset.pixel_array)
    _max = np.min(dataset.pixel_array)
    return _id, _age, _sex, _min, _max, _mean

responses = []
for file_name in tqdm(os.listdir('../input/stage_1_train_images/')):
    
    file_path = os.path.join('../input/stage_1_train_images/', file_name)
    dataset = pydicom.dcmread(file_path)

    responses.append(pool.apply_async(process_image, (dataset,)))


pool.close()
pool.join()
for response in tqdm(responses):
    _id, _age, _sex, _min, _max, _mean = response.get()
    ids.append(_id)
    ages.append(_age)
    sexs.append(_sex)
    img_min_lums.append(_min)
    img_max_lums.append(_max)
    img_avg_lums.append(_mean)


train_demo_df['patientId'] = pd.Series(ids)
train_demo_df['patientAge'] = pd.Series(ages, dtype='int')
train_demo_df['patientSex'] = pd.Series(sexs)

train_demo_df['imageMin'] = pd.Series(img_max_lums)
train_demo_df['imageMax'] = pd.Series(img_min_lums)
train_demo_df['imageMean'] = pd.Series(img_avg_lums)

sex_map = {'F': 0, 'M': 1}
train_demo_df['patientSex'] = train_demo_df['patientSex'].replace(sex_map).astype('int')
class_df = pd.read_csv('../input/stage_1_detailed_class_info.csv')

train_demo_df = pd.merge(left=train_demo_df, right=class_df, left_on='patientId', right_on='patientId')
print(train_demo_df.describe())
train_demo_df.head()
for file_name in tqdm(os.listdir('../input/stage_1_train_images/')):
    file_path = os.path.join('../input/stage_1_train_images/', file_name)
    dataset = pydicom.dcmread(file_path)
    if int(dataset.PatientAge) >= 100:
        show_dcm_info(dataset)
        plot_pixel_array(dataset)
train_demo_df = train_demo_df.where(train_demo_df['patientAge'] <= 100, train_demo_df['patientAge']-100, axis=1)
sns.pairplot(train_demo_df, hue='class', height=3)
