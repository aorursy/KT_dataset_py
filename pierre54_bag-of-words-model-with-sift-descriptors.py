import cv2

import numpy as np

import os

import pandas as pd

import csv



from sklearn.cluster import MiniBatchKMeans

from sklearn.neural_network import MLPClassifier
img_path = '../input/images/'

train = pd.read_csv('../input/train.csv')

species = train.species.sort_values().unique()



dico = []



def step1():

    for leaf in train.id:

        img = cv2.imread(img_path + str(leaf) + ".jpg")

        kp, des = sift.detectAndCompute(img, None)



        for d in des:

            dico.append(d)
def step2():

    k = np.size(species) * 10



    batch_size = np.size(os.listdir(img_path)) * 3

    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1).fit(dico)
def step3():

    kmeans.verbose = False



    histo_list = []



    for leaf in train.id:

        img = cv2.imread(img_path + str(leaf) + ".jpg")

        kp, des = sift.detectAndCompute(img, None)



        histo = np.zeros(k)

        nkp = np.size(kp)



        for d in des:

            idx = kmeans.predict([d])

            histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly



        histo_list.append(histo)
def step4():

    X = np.array(histo_list)

    Y = []



    # It's a way to convert species name into an integer

    for s in train.species:

        Y.append(np.min(np.nonzero(species == s)))



    mlp = MLPClassifier(verbose=True, max_iter=600000)

    mlp.fit(X, Y)
def step5():

    test = pd.read_csv('../input/test.csv')



    result_file = open("sift.csv", "w")

    result_file_obj = csv.writer(result_file)

    result_file_obj.writerow(np.append("id", species))



    for leaf in test.id:

        img = cv2.imread(img_path + str(leaf) + ".jpg")

        kp, des = sift.detectAndCompute(img, None)



        x = np.zeros(k)

        nkp = np.size(kp)



        for d in des:

            idx = kmeans.predict([d])

            x[idx] += 1/nkp



        res = mlp.predict_proba([x])

        row = []

        row.append(leaf)



        for e in res[0]:

            row.append(e)



        result_file_obj.writerow(row)



    result_file.close()