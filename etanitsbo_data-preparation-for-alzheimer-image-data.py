

import numpy as np 

import pandas as pd

import os as os





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# just quick result for
def readLabels():

    

    udir = "../input"    

    

    filename = "adni_demographic_master_kaggle.csv"

    fullpath = os.path.join(udir,filename)

    

    demo = pd.read_csv(fullpath)

    print("-- demographic data : \n")

    print(demo.head())

    

    trX_subjs = demo[(demo['train_valid_test']==0)]

    trY_diagnosis = np.asarray(trX_subjs.diagnosis)



    vaX_subjs = demo[(demo['train_valid_test']==1)]

    vaY_diagnosis = np.asarray(vaX_subjs.diagnosis)



    train_orig = trY_diagnosis

    valid_orig = vaY_diagnosis    

    

    images_per_sub = 62



    """

    diagnosis x 62 == total images of one Subjects    

        

    """

    trY_all = []

    for n in trY_diagnosis:

        for i in range(images_per_sub):

            trY_all.append(n-1)



    trY_all = np.asarray(trY_all)

    

    vaY_all = []

    for n in vaY_diagnosis:

        for i in range(images_per_sub):

            vaY_all.append(n-1)

        

    vaY_all = np.asarray(vaY_all)

            

    """        

    trY_targets = np.zeros((len(trY_all), 3))

    for count, target in enumerate(trY_all):

        trY_targets[count][ target - 1 ]= 1    

    print(trY_targets)

    """

        

    #trainingOneHot =LabelBinarizer().fit_transform(trY_all)

    #validOneHot =LabelBinarizer().fit_transform(vaY_all)

    #print "-- length for traing / valid target flag",  len(trainingOneHot), len(validOneHot)

    

    return trY_all,vaY_all

    

training_labels, var_labels = readLabels()



len(training_labels), len(var_labels)

list(set(training_labels)),list(set(var_labels))
def alzImageData():

    

    # trainign directory index

    # ima

    training_idx = range(1,3)



    fileslist = []    

    for idx in training_idx:

        print(".... reading imgset dirctory: imgset_%d" % idx) 

               

        alzDir = "../input"

        udir = alzDir + "/imgset_%d/imgset_%d" % (idx,idx)

        files = os.listdir(udir)

        #

        trainings = [  os.path.join(udir,  f  )   for f in files if f[-3:] == 'npy' ]

        print(trainings)

        fileslist.extend(trainings)



    return fileslist



files = alzImageData()

sorted(files)
def StackImages(files):



    prev_data = np.array([])

    s_image = np.array([])

    

    for idx, f in enumerate(files):

        print("- reading image npy files ...", f)

        img_data = np.load(f)

        

        if idx > 0:

            print("-- images stacked No. %d" % (idx+1))

            s_image = np.vstack((prev_data,img_data))

            prev_data = s_image

        else:

            prev_data = img_data

                    

    return s_image    



images = StackImages(sorted(files))

print(images.shape)

l,h,w = images.shape

r = np.random.permutation(l)

print(" -- random index :", r)

print("-- just pick first item for inspecting image", r[0])
import matplotlib.pyplot as plt

import cv2



inspect_image = images[r[0]]

print(inspect_image.shape)



plt.hist(inspect_image)

plt.show()

plt.imshow(inspect_image,cmap='Greys',  interpolation='nearest')

plt.show()





inspect_image *= 255.0/inspect_image.max()

plt.hist(inspect_image)

plt.show()

plt.imshow(inspect_image,cmap='Greys',  interpolation='nearest')

plt.show()



size_ = (227,227)

changed_image_ = cv2.resize(inspect_image,size_)



print("-- new sized shape:", changed_image_.shape)



plt.hist(inspect_image)

plt.show()

plt.imshow(inspect_image,cmap='Greys',  interpolation='nearest')

plt.show()






