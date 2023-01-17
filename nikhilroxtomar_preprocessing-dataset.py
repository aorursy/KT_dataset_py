import numpy as np
import pydicom
import cv2
import os
from tqdm import tqdm
import pandas as pd
!ls ../input
test_path = '../input/stage_1_test_images/'
train_path = '../input/stage_1_train_images/'
test_save_path = '../working/test_image/'
train_save_path = '../working/train_image/'
def save_image(img, file_path):
    """
    :param img - numpy array (image)
    :param file_path - path where to save file
    """
    ## You can even add a cv2.resize() function to resize the image
    cv2.imwrite(file_path, img)
def read_dcm_file(file_path):
    """
    :param file_path - path of the DCM image file
    """
    dcm_file = pydicom.read_file(file_path)
    return dcm_file
def main(path, name, save_path):
    """
    :param path - path for the DCM image files
    :param name - name of the csv file
    :param save_path - path where to save the image
    """

    list_files = os.listdir(path)
    #['PatientAge', 'PatientBirthDate', 'PatientID', 'PatientName', 'PatientOrientation', 'PatientSex']

    p_age_list = []
    p_birthdate_list = []
    p_id_list = []
    p_name_list = []
    p_orientation_list = []
    p_sex_list = []

    for idx, file_name in tqdm(enumerate(list_files)):
        file_id = file_name.split('.')[0]
        file_path = path + file_name
        dcm_data = read_dcm_file(file_path)
        #dcm_attr = dcm_data.dir('pat')
        image = dcm_data.pixel_array
        image_path = save_path + file_id + '.png'
        save_image(image, image_path)

        p_id = dcm_data.PatientID
        p_name = dcm_data.PatientName
        p_age = dcm_data.PatientAge
        p_sex = dcm_data.PatientSex
        p_orientation = dcm_data.PatientOrientation

        p_id_list.append(p_id)
        p_name_list.append(p_name)
        p_age_list.append(p_age)
        p_sex_list.append(p_sex)
        p_orientation_list.append(p_orientation)

    df = pd.DataFrame()
    df['patientId'] = p_id_list
    df['patientName'] = p_name_list
    df['patientAge'] = p_age_list
    df['patientSex'] = p_sex_list
    df['patientOrientation'] = p_orientation_list

    df.to_csv(name)
def parse_train_label(df, file_name):
    """
    Method to read a CSV file (Pandas dataframe) and parse the
    data into the following nested dictionary:

      parsed = {

        'patientId-00': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia,
            'boxes': list of box(es)
        },
        'patientId-01': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia,
            'boxes': list of box(es)
        }, ...

      }

    """
    # --- Define lambda to extract coords in list [y, x, height, width]
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]

    parsed = {}
    for n, row in df.iterrows():
        # --- Initialize patient entry into parsed
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'patientId': pid,
                'label': row['Target'],
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    s = pd.DataFrame.from_dict(parsed, orient='index', columns=['patientId', 'label', 'boxes'])
    s.to_csv(file_name)
def parse_class_info(df, file_name):
    parsed = {}
    for n, row in df.iterrows():
        pid = row['patientId']

        if pid not in parsed:
            parsed[pid]  = {
                'patientId': pid,
                'class': [row['class']]
            }
        else:
            parsed[pid]['class'].append(row['class'])

    s = pd.DataFrame.from_dict(parsed, orient='index', columns=['patientId', 'class'])
    s.to_csv(file_name)
## Train DCM Images
main(train_path, '../working/train_image.csv', train_save_path)
## Test DCM Images
main(test_path, '../working/test_image.csv', test_save_path)

## Combining all the data
train_label = pd.read_csv('../input/stage_1_train_labels.csv')
parse_train_label(train_label, '../working/train_labels.csv')

class_info = pd.read_csv('../input/stage_1_detailed_class_info.csv')
parse_class_info(class_info, '../working/class_info.csv')

train_label = pd.read_csv('../working/train_labels.csv', index_col='patientId')
class_info = pd.read_csv('../working/class_info.csv', index_col='patientId')
train_image = pd.read_csv('../working/train_image.csv', index_col='patientId')

train_label['class'] = class_info['class']
train_label['age'] = train_image['patientAge']
train_label['sex'] = train_image['patientSex']
train_label.to_csv('final_train_data.csv')

import base64
import pandas as pd
from IPython.display import HTML

def create_download_link( df, title = "Download CSV file", filename = "download_final_train_data.csv"):
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(train_label)
