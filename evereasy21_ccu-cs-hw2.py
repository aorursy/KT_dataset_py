import numpy as np

import scipy.io as sio 

import os

import cv2

import matplotlib.pyplot as plt

import pprint

import zipfile
def get_labels(path):

    annos = sio.loadmat('../input/devkit/devkit/' + path )

    _, total_size = annos["annotations"].shape

    print("total sample size is ", total_size)

    labels = np.zeros((total_size, 5))

    for i in range(total_size):

        path = annos["annotations"][:,i][0][5][0].split(".")

        id = int(path[0]) - 1

        for j in range(5):

            labels[id, j] = int(annos["annotations"][:,i][0][j][0])

    return labels
def read_data(path, labels):

    counter = 0

    

    for file in os.listdir("../input/" + path + "/" + path):

#         im = cv2.imread("../input/" + path + "/" + path + "/" + file)[:,:,::-1]

        im = cv2.imread("../input/" + path + "/" + path + "/" + file)

        name = file.split('.')

        image_label = labels[int(name[0]) - 1]

        x = (im[int(image_label[1]):int(image_label[3]),int(image_label[0]):int(image_label[2])])

        y = (int(image_label[4]))

        if counter % 1000 == 0 and counter > 0:

            print("1000 images are loaded.")

        counter += 1

        if not os.path.exists('./'+path+'/'+ str(y)):

            os.makedirs(path+'/'+ str(y))

        cv2.imwrite(path+'/'+ str(y) +"/" + file, x)

        

train_labels = get_labels('cars_train_annos.mat')

read_data("cars_train", train_labels)

print("Cutting training data completed.")
test_labels = get_labels('cars_test_annos_withlabels.mat')

read_data("cars_test", test_labels)

print("Cutting testing data completed.")
def zipdir(path, ziph):

    # ziph is zipfile handle

    for root, dirs, files in os.walk(path):

        for file in files:

            ziph.write(os.path.join(root, file))
zipf = zipfile.ZipFile('cars_train.zip', 'w', zipfile.ZIP_DEFLATED)

zipdir('cars_train', zipf)

zipf.close()
zipf = zipfile.ZipFile('cars_test.zip', 'w', zipfile.ZIP_DEFLATED)

zipdir('cars_test', zipf)

zipf.close()
!rm -rf cars_train

!rm -rf cars_test