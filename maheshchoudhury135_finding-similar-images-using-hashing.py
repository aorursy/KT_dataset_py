#Installation

!pip install vptree

!pip install imutils
#Importing dependencies

import numpy as np

import pandas as pd

import cv2

from imutils import paths

import argparse

import pickle

import vptree

import matplotlib.pyplot as plt

import time
#For Image Hashing

def dhash(image, hashSize=8):

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	resized = cv2.resize(gray, (hashSize + 1, hashSize))

	diff = resized[:, 1:] > resized[:, :-1]

	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])



#Converting data types

def convert_hash(h):

	return int(np.array(h, dtype="float64"))



def hamming(a, b):

	return bin(int(a) ^ int(b)).count("1")
imagePaths = list(paths.list_images("../input/image-for-similarity/image_Similarity/"))

hashes= {}



#Loading and hashing the images

for (i, img_path) in enumerate(imagePaths):

    

    image = cv2.imread(img_path)

    h = dhash(image)

    h = convert_hash(h)

    l = hashes.get(h, [])

    l.append(img_path)

    hashes[h] = l
points = list(hashes.keys())

tree = vptree.VPTree(points, hamming) #Creating VP-Tree
#serializing VP-Tree

f = open("tree.pickle", "wb")

f.write(pickle.dumps(tree))

f.close()



# serialize the hashes to dictionary

f = open("hashes.pickle", "wb")

f.write(pickle.dumps(hashes))

f.close()
tree = pickle.loads(open("tree.pickle", "rb").read())

hashes = pickle.loads(open("hashes.pickle", "rb").read())



# compute the hash for the query image, then convert it

queryHash = dhash(image)

queryHash = convert_hash(queryHash)
image = cv2.imread("../input/image-for-similarity/image_Similarity/airplanes/image_0001.jpg")
plt.imshow(image)
queryHash = dhash(image)

queryHash = convert_hash(queryHash)
start = time.time()

results = tree.get_all_in_range(queryHash, 13)

results = sorted(results)

end = time.time()

print(">>> search took {} seconds".format(end - start))
# loop over the results

for (d, h) in results[:2]:

    resultPaths = hashes.get(h, [])

    print(">>> {} total image(s) with d: {}, h: {}".format(len(resultPaths), d, h))

    for resultPath in resultPaths:

        result = cv2.imread(resultPath)

        plt.imshow(result)

        plt.show()