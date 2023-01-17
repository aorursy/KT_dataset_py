import pydicom

import os

import numpy as np

import pandas as pd

from matplotlib import cm

from matplotlib import pyplot as plt

import cv2

import seaborn as sns

from tqdm import tqdm

import sys

import SimpleITK as sitk

import numpy as np

import nibabel as nib

import os

import itertools

from sklearn.metrics import f1_score

import xlwt 

from xlwt import Workbook 

num_of_slice = 0

for file_name in os.listdir('../input/pat02dicomsmask'):

    file_path = os.path.join('../input/pat02dicomsmask', file_name)

    num_of_slice += 1

    wb = Workbook()

i = 0

num_to_plot = num_of_slice  # numero su cui lavorare

truth_mask = np.zeros((num_of_slice,630,630))



for file_name in sorted(os.listdir('../input/pat02dicomsmask')):

    file_path = os.path.join('../input/pat02dicomsmask', file_name)

    ds = pydicom.filereader.dcmread(file_path)

    #print(file_name)



    truth_mask[i] = np.array(ds.pixel_array)

    i += 1



    if i == num_to_plot:

        break

    
print(truth_mask[30])

print(truth_mask.shape)



plt.imshow(truth_mask[30])

plt.show()
num_of_slice = 0

for file_name in os.listdir('../input/pat02dicomsmaskr231'):

    file_path = os.path.join('../input/pat02dicomsmaskr231', file_name)

    num_of_slice += 1



i = 0

num_to_plot =  num_of_slice # numero su cui lavorare

dnn_mask = np.zeros((num_of_slice,630,630))



for file_name in sorted(os.listdir('../input/pat02dicomsmaskr231')):

    file_path = os.path.join('../input/pat02dicomsmaskr231', file_name)

    ds = pydicom.filereader.dcmread(file_path)

    #print(file_name)



    dnn_mask[i] = np.array(ds.pixel_array)

    i += 1



    if i == num_to_plot:

        break

        
for i in range (0, num_of_slice):

    dnn_mask[i] = np.where(dnn_mask[i] == 2, 1, dnn_mask[i]) 



print(dnn_mask.shape)

plt.imshow(dnn_mask[30])

plt.show()



#np.set_printoptions(threshold=sys.maxsize) #stampare matrice completa


sheet1 = wb.add_sheet('Sheet 1', cell_overwrite_ok=True) 

style = xlwt.easyxf('font: bold 1, color red;')



sheet1.write(0, 0, "R231", style) 

sheet1.write(1, 0, "IoU_Pat2", style) 

sheet1.write(1, 1, "Dice_Pat2", style) 



for i in range (0, num_of_slice):

    

    '''Intersection over Union (IoU) '''

    intersection = np.logical_and(truth_mask[i], dnn_mask[i])

    union = np.logical_or(truth_mask[i], dnn_mask[i])

    iou_score = np.sum(intersection) / np.sum(union)

    sheet1.write(i+2, 0, iou_score) 

    #print("The slice ",i," has IoU = ", iou_score)

    

    

    '''DICE Coefficient (F1 Score)'''

    sheet1.write(i+2, 1, f1_score(truth_mask[i], dnn_mask[i], zero_division=0, average='micro')) 

    #f1_score(truth_mask[30], dnn_mask[30], average='micro')

    #f1_score(truth_mask[30], dnn_mask[30], average='macro')

    

wb.save('model_Pat2.xls')









num_of_slice = 0

for file_name in os.listdir('../input/pat02dicomsmaskcovidweb'):

    file_path = os.path.join('../input/pat02dicomsmaskcovidweb', file_name)

    num_of_slice += 1



i = 0

num_to_plot =  num_of_slice # numero su cui lavorare

dnn_mask = np.zeros((num_of_slice,630,630))



for file_name in sorted(os.listdir('../input/pat02dicomsmaskcovidweb')):

    file_path = os.path.join('../input/pat02dicomsmaskcovidweb', file_name)

    ds = pydicom.filereader.dcmread(file_path)

    #print(file_name)



    dnn_mask[i] = np.array(ds.pixel_array)

    i += 1



    if i == num_to_plot:

        break
for i in range (0, num_of_slice):

    dnn_mask[i] = np.where(dnn_mask[i] == 2, 1, dnn_mask[i]) 



print(dnn_mask.shape)

plt.imshow(dnn_mask[30])

plt.show()



#np.set_printoptions(threshold=sys.maxsize) #stampare matrice completa
def dice(img1, img2):



    img1 = np.asarray(img1).astype(np.bool)

    img2 = np.asarray(img2).astype(np.bool)

    if img1.sum() + img2.sum() == 0: return 1

    intersection = np.logical_and(img1, img2)

    return 2. * intersection.sum() / (img1.sum() + img2.sum())

sheet1.write(0, 3, "CovidWeb", style) 

sheet1.write(1, 3, "IoU_Pat2", style) 

sheet1.write(1, 4, "Dice_Pat2", style) 



for i in range (0, num_of_slice):

    

    '''Intersection over Union (IoU) '''

    intersection = np.logical_and(truth_mask[i], dnn_mask[i])

    union = np.logical_or(truth_mask[i], dnn_mask[i])

    iou_score = np.sum(intersection) / np.sum(union)

    sheet1.write(i+2, 3, iou_score) 

    #print("The slice ",i," has IoU = ", iou_score)

    

    

    '''DICE Coefficient (F1 Score)'''

    sheet1.write(i+2, 4, f1_score(truth_mask[i], dnn_mask[i], zero_division=0, average='micro')) 

    #f1_score(truth_mask[30], dnn_mask[30], average='micro')

    #f1_score(truth_mask[30], dnn_mask[30], average='macro')

    sheet1.write(i+2, 7, dice(truth_mask[i], dnn_mask[i]))

wb.save('model_Pat2.xls')


