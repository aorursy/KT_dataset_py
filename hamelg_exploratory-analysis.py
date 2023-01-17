import dicom

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os
print( os.listdir("../input/") )
patients = os.listdir("../input/sample_images")

print(patients)

print(len(patients))
patient_dict = {}



for patient in patients:

    patient_dict[patient] = os.listdir("../input/sample_images/" + patient)



print(patient_dict['0d941a3ad6c889ac451caf89c46cb92a'][0])

print(len(patient_dict['0d941a3ad6c889ac451caf89c46cb92a']))
for patient in patient_dict.values():

    print (len( patient ))
image_dict = {}



for patient, images in patient_dict.items():

    image_dict[patient] = []

    for image in images:

        img = dicom.read_file("../input/sample_images/" + patient + "/" + image)

        image_dict[patient].append(img)

        

image_dict[patients[0]][0]
for patient, images in image_dict.items():

    num_images = len(images)

    instances = []

    for image in images:

        instances.append(image.InstanceNumber)

    instances = set(sorted(instances))

    if len(instances) != num_images:

        print("Unique instances and number of images are not equal!")



# Inspect the instance numbers of the last patient

print(instances)
for p, imgs in image_dict.items():

    image_dict[p] = sorted(imgs, key=lambda x: x.InstanceNumber)
feature_names = image_dict[patients[0]][0].dir()



for p in range(20):

    f_names = image_dict[patients[p]][0].dir()

    if feature_names == f_names:

        print("Features equal!")

    else:

        print("Features NOT equal!")
feature_names = set(image_dict[patients[0]][0].dir())



for p in range(1, 20):

    f_names = set(image_dict[patients[p]][0].dir())

    feature_names = feature_names.intersection(f_names)



feature_names.remove("PixelData")



print(feature_names)

print(len(feature_names))
img_feats = []



for p, imgs in image_dict.items():

    for img in imgs:

        feats = [img.data_element(f).value for f in feature_names]

        img_feats.append(feats)

        

feature_frame = pd.DataFrame(img_feats, columns=feature_names)
feature_frame.head()
print(feature_frame.WindowWidth.head())

print(feature_frame.WindowWidth.tail())

print(feature_frame.WindowCenter.head())

print(feature_frame.WindowCenter.tail())
feature_frame.drop('WindowWidth', inplace=True, axis=1)

feature_frame.drop('WindowCenter', inplace=True, axis=1)
for col in feature_frame.columns:

    if type(feature_frame[col][0]) == dicom.multival.MultiValue:

        print(col)

        multi_len = len(feature_frame[col][0])

        for val in range(multi_len):

            feature_frame[col + str(val)] = [feature_frame[col][x][val] for x in range(len(feature_frame))]

        feature_frame.drop(col, inplace=True, axis=1)
unhashable = [dicom.multival.MultiValue, dicom.valuerep.PersonName3]



for col in feature_frame.columns:

    if type(feature_frame[col][0]) not in unhashable:

        if len(feature_frame[col].unique()) == 1:

            feature_frame.drop(col, inplace=True, axis=1)

            print(col, "dropped!")
for col in feature_frame.columns:

    if "UID" in col:

        feature_frame.drop(col, inplace=True, axis=1)

        print(col, "dropped!")

        

# Drop repeated column ('PatientName' since it is the same as 'PatientID')

feature_frame.drop('PatientName', inplace=True, axis=1)
print(len([f for f in feature_frame.PositionReferenceIndicator if f == '']))



feature_frame.drop('PositionReferenceIndicator', inplace=True, axis=1)
for col in feature_frame.columns:

    print(type(feature_frame[col][0]))
feature_frame.describe()
feature_frame.corr()
feature_frame.to_csv("metadata_features.csv")
feature_frame.to_csv("metadata_features.csv")