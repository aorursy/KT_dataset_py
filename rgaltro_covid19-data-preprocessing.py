# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import shutil

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
metadata = pd.read_csv('/kaggle/input/covid19-cases/metadata.csv')
metadata.columns
for idx, data in metadata.groupby('sex'):

    print(idx, data.shape)
import matplotlib.pyplot as plt

list_age = []

number_cases = []

for idx, data in metadata.groupby('age'):

    list_age.append(idx)

    number_cases.append(data.shape[0])
plt.bar(list_age, number_cases)
list_of_covid19_temp = []

other_than_covid_temp = []

for idx, data in metadata.groupby('finding'):

    print(f'The patients having {idx} are {data.shape[0]}')

    if idx == 'COVID-19' or idx == 'COVID-19, ARDS':

        list_of_covid19_temp.append(list(data.filename.values))

    else:

        other_than_covid_temp.append(list(data.filename.values))

list_of_covid19 = [y for x in list_of_covid19_temp for y in x]

other_than_covid = [y for x in other_than_covid_temp for y in x]
os.makedirs('final_dataset')
os.makedirs('final_dataset/train/COVID19')

os.makedirs('final_dataset/train/NORMAL')

os.makedirs('final_dataset/train/OTHERS')

os.makedirs('final_dataset/test/COVID19')

os.makedirs('final_dataset/test/NORMAL')

os.makedirs('final_dataset/test/OTHERS')

os.makedirs('final_dataset/val/COVID19')

os.makedirs('final_dataset/val/NORMAL')

os.makedirs('final_dataset/val/OTHERS')
import random

list_of_covid19_copy = list_of_covid19.copy()
overall_test_images_covid = random.sample(list_of_covid19, 30)

test_images_covid = random.sample(overall_test_images_covid,23)

val_images_covid = [x for x in overall_test_images_covid if x not in test_images_covid]
train_images_covid = [x for x in list_of_covid19 if x not in overall_test_images_covid]
overall_test_images_other = random.sample(other_than_covid, 10)

test_images_other = random.sample(overall_test_images_other,7)

val_images_other = [x for x in overall_test_images_other if x not in test_images_other]

train_images_other = [x for x in other_than_covid if x not in overall_test_images_other]
covid19_data_list = [train_images_covid, test_images_covid, val_images_covid]
for idx, data in enumerate(covid19_data_list):

    if idx == 0:

        for img_name in data:

            try:

                shutil.copy(os.path.join("/kaggle/input/covid19-cases/images", img_name),

                        os.path.join("final_dataset/train/COVID19/",img_name))

            except:

                train_images_covid.remove(img_name)

    elif idx == 1:

        for img_name in data:

            try:

                shutil.copy(os.path.join("/kaggle/input/covid19-cases/images", img_name),

                        os.path.join("final_dataset/test/COVID19/",img_name))

            except:

                test_images_covid.remove(img_name)

    else:

        for img_name in data:

            try:

                shutil.copy(os.path.join("/kaggle/input/covid19-cases/images", img_name),

                        os.path.join("final_dataset/val/COVID19/",img_name))

            except:

                val_images_covid.remove(img_name)
len(train_images_covid), len(test_images_covid), len(val_images_covid)
os.makedirs('label_file/train')

os.makedirs('label_file/test')

os.makedirs('label_file/val')
np.save('label_file/train/covid19.npy', train_images_covid)

np.save('label_file/test/covid19.npy', test_images_covid)

np.save('label_file/val/covid19.npy', val_images_covid)
other_images_list = [train_images_other, test_images_other, val_images_other]

for idx, data in enumerate(other_images_list):

    if idx == 0:

        for img_name in data:

            try:

                shutil.copy(os.path.join("/kaggle/input/covid19-cases/images", img_name),

                        os.path.join("final_dataset/train/OTHERS/",img_name))

            except:

                train_images_other.remove(img_name)

    elif idx == 1:

        for img_name in data:

            try:

                shutil.copy(os.path.join("/kaggle/input/covid19-cases/images", img_name),

                        os.path.join("final_dataset/test/OTHERS/",img_name))

            except:

                test_images_other.remove(img_name)

    else:

        for img_name in data:

            try:

                shutil.copy(os.path.join("/kaggle/input/covid19-cases/images", img_name),

                        os.path.join("final_dataset/val/OTHERS/",img_name))

            except:

                val_images_other.remove(img_name)
len(train_images_other), len(test_images_other), len(val_images_other)
np.save('label_file/train/other.npy', train_images_other)

np.save('label_file/test/other.npy', test_images_other)

np.save('label_file/val/other.npy', val_images_other)
train_pne_normal = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/')

test_pne_normal = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL/')

val_pne_normal = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/val/NORMAL/')

train_pne = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/')

test_pne = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/')

val_pne = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/')
np.save('label_file/train/normal.npy', train_pne_normal)

np.save('label_file/test/normal.npy', test_pne_normal)

np.save('label_file/val/normal.npy', val_pne_normal)

np.save('label_file/train/pne.npy', train_pne)

np.save('label_file/test/pne.npy', test_pne)

np.save('label_file/val/pne.npy', val_pne)
overall_pne_list = [train_pne_normal, train_pne, test_pne_normal, test_pne, val_pne_normal, val_pne]
for idx, data in enumerate(overall_pne_list):

    print(idx)

    if idx == 0:

        for img_name in data:

            shutil.copy(os.path.join("/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL", img_name),

                        os.path.join("final_dataset/train/NORMAL/",img_name))

    elif idx == 1:

        for img_name in data:

            shutil.copy(os.path.join("/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/", img_name),

                        os.path.join("final_dataset/train/OTHERS/",img_name))

    elif idx == 2:

        for img_name in data:

            shutil.copy(os.path.join("/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL", img_name),

                        os.path.join("final_dataset/test/NORMAL/",img_name))

    elif idx == 3:

        for img_name in data:

            shutil.copy(os.path.join("/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/", img_name),

                        os.path.join("final_dataset/test/OTHERS/",img_name))

    elif idx == 4:

        for img_name in data:

            shutil.copy(os.path.join("/kaggle/input/chest-xray-pneumonia/chest_xray/val/NORMAL", img_name),

                        os.path.join("final_dataset/val/NORMAL/",img_name))

    elif idx == 5:

        for img_name in data:

            shutil.copy(os.path.join("/kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/", img_name),

                        os.path.join("final_dataset/val/OTHERS/",img_name))
!tar chvfz notebook_updated.tar.gz *
len(val_pne)
os.remove('notebook.tar.gz')
os.remove('notebook_updated.tar.gz')