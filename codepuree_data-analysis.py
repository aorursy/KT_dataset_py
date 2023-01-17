# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from os import path as path

import os



dir_data = '../input/train-jpg'



list_img_names = os.listdir(dir_data)



print('Number of chips in folder {}: {}'.format(dir_data, len(list_img_names)))

print('Sample from list_img_names: {}'.format(list_img_names[:5]))
labels_path = '../input/train_v2.csv'

lables_file = open(labels_path, 'r')



lines = lables_file.readlines()



print('The file contains {} lines.'.format(len(lines)))



print('Head: {}'.format(lines[0]))

print('Sample Line: {}'.format(lines[1]))



labels = dict()

for i in range(1, len(lines)):

    line = lines[i].replace('\n', '')

    elements = line.split(',')

    labels[elements[0]] = elements[1].split(' ')
import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import cv2

import math



sample_display_range = 10

lin = list_img_names[-20:]

plt.figure()#figsize=(40, 40))

for i in range(10):#sample_display_range):

    plt.figure()

    im_name_ext = lin[i]

    im_path = path.join(dir_data, im_name_ext)

    im_name = im_name_ext.split('.')[0]

    im = cv2.imread(im_path, 1)

    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)



    h,w = im.shape[:2]

    #plt.subplot(math.ceil(sample_display_range / 5), 5, i+1)

    plt.imshow(im)

    #plt.axis('off')

    plt.title(im_name)

    plt.xlabel(''.join(str(item + ' ') for item in labels[im_name]))

plt.show()
diff_labels = []



for inLabels in labels:

    for label in labels[inLabels]:

        if label not in diff_labels:

            diff_labels.append(label)

            

print('The dataset contains {} different labels:'.format(len(diff_labels)))

print(diff_labels)
comb_labels = dict()



# find all combinations and count them

for im_name in labels:

    im_labels = labels[im_name]

    im_labels.sort()

    im_label = ' '.join(im_labels)

    

    if im_label in comb_labels:

        comb_labels[im_label] += 1

    else:

        comb_labels[im_label] = 1



# sort combinations by number of occurences

comb_labels = dict(sorted(comb_labels.items(), key=lambda x: -x[1]))



# print statitics

print('There are {} different combinations of labels:'.format(len(comb_labels)))

for label in comb_labels:

    print('\t{}:\t{}'.format(comb_labels[label], label))
cloud_coverage = dict({'haze': [], 'clear': [], 'partly_cloudy': [], 'cloudy': []})

unknown_cloud_coverage = []

unknown_labels = []



for im_name_ext in list_img_names:

    im_name = im_name_ext.split('.')[0]

    

    if im_name in labels:

        if 'haze' in labels[im_name]:

            cloud_coverage['haze'].append(im_name)

        elif 'clear' in labels[im_name]:

            cloud_coverage['clear'].append(im_name)

        elif 'partly_cloudy' in labels[im_name]:

            cloud_coverage['partly_cloudy'].append(im_name)

        elif 'cloudy' in labels[im_name]:

            cloud_coverage['cloudy'].append(im_name)

        else:

            unknown_cloud_coverage.append(im_name)

    else:

        unknown_labels.append(im_name)

        

# Print cloud coverage statistics

total_cloud_coverage = 0

print('The dataset contains:')

for label in cloud_coverage:

    print('\t{}:\t{}'.format(label.replace('_', ' '), len(cloud_coverage[label])))

    total_cloud_coverage += len(cloud_coverage[label])



print('''The dataset contain {} images, that do not have a lable and {} images that do not have a

cloud coverage label.'''.format(len(unknown_labels), len(unknown_cloud_coverage)))



print('''\nThe cloud_coverage datasets contains {} images in total, 

that are {} less then all images.'''.format(total_cloud_coverage, (len(labels) - total_cloud_coverage)))



print('''\nThis are the images that do not contain any cloud coverage label.''')

for im_name in unknown_cloud_coverage:

    print('\t{}:\t{}'.format(im_name, ' '.join([(label.replace('_', ' ')) for label in labels[im_name]])))
ground_coverage = dict({'agriculture': [], 'primary': [], 'cultivation': [], 'road': [], 'water': [], 'selective_logging': [],

                        'bare_ground': [], 'habitation': [], 'artisinal_mine': [], 'blooming': [], 'blow_down': [],

                        'slash_burn': [], 'conventional_mine': []})



for im_name_ext in list_img_names:

    im_name = im_name_ext.split('.')[0]

    

    if im_name in labels:

        im_labels = labels[im_name]

        

        for label in im_labels:

            if label in ground_coverage:

                ground_coverage[label].append(im_name)



# Print cloud coverage statistics

print('The dataset contains:')

for label in ground_coverage:

    print('\t{}:\t{}'.format(label.replace('_', ' '), len(ground_coverage[label])))
from copy import copy



sample_haze_img_name = cloud_coverage['haze'][1337]



sample_haze_img_path = os.path.join('../input/train-jpg', sample_haze_img_name + '.jpg')



plt.figure()



sample_haze_img = cv2.imread(sample_haze_img_path, 1)

sample_haze_img = cv2.cvtColor(sample_haze_img, cv2.COLOR_BGR2RGB)



h,w = sample_haze_img.shape[:2]



plt.subplot(1, 3, 1)

plt.imshow(sample_haze_img)



plt.title(sample_haze_img_name)

plt.xlabel(''.join(str(item + ' ') for item in labels[sample_haze_img_name]))



plt.subplot(1, 3, 2)

plt.hist(sample_haze_img.ravel(),256,[0,256]);



plt.subplot(1, 3, 3)

sample_haze_equ = cv2.equalizeHist(cv2.cvtColor(sample_haze_img, cv2.COLOR_BGR2GRAY))

plt.imshow(sample_haze_equ, cmap='gray')



# dehaze sample image



# find darkest pixel in each channle and substract it from all other

minRGB = [99999999, 99999999, 99999999]



for row in sample_haze_img:

    for pixel in row:

        minRGB[0] = min(minRGB[0], pixel[0])

        minRGB[1] = min(minRGB[1], pixel[1])

        minRGB[2] = min(minRGB[2], pixel[2])

        

print('minRGB:\t{}'.format(minRGB))



sample_dehaze_img = copy(sample_haze_img)



# dehaze image

for row in sample_dehaze_img:

    for pixel in row:

        pixel[0] = pixel[0] - minRGB[0]

        pixel[1] = pixel[1] - minRGB[1]

        pixel[2] = pixel[2] - minRGB[2]

        

print('haze:\t{}\t|\tdehaze:\t{}'.format(sample_haze_img[1][1], sample_dehaze_img[1][1]))



plt.figure()



h,w = sample_dehaze_img.shape[:2]



plt.subplot(1, 3, 1)

plt.imshow(sample_dehaze_img)



plt.title('dehazed {}'.format(sample_haze_img_name))

plt.xlabel(''.join(str(item + ' ') for item in labels[sample_haze_img_name]))



plt.subplot(1, 3, 2)

plt.hist(sample_dehaze_img.ravel(),256,[0,256]);



plt.subplot(1, 3, 3)

sample_dehaze_equ = cv2.equalizeHist(cv2.cvtColor(sample_dehaze_img, cv2.COLOR_BGR2GRAY))

plt.imshow(sample_dehaze_equ, cmap='gray')