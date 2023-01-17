import numpy as np

import csv

import cv2

import glob

import os

import xml.etree.ElementTree as ET
SPLIT_RATIO = 0.8

XMLS = "../input/annotations/annotations/xmls"
class_names = {}

k = 0

output = []

xml_files = glob.glob("{}/*xml".format(XMLS))

for i, xml_file in enumerate(xml_files):

    tree = ET.parse(xml_file)



    path = os.path.join(XMLS, tree.findtext("./filename"))



    height = int(tree.findtext("./size/height"))

    width = int(tree.findtext("./size/width"))

    xmin = int(tree.findtext("./object/bndbox/xmin"))

    ymin = int(tree.findtext("./object/bndbox/ymin"))

    xmax = int(tree.findtext("./object/bndbox/xmax"))

    ymax = int(tree.findtext("./object/bndbox/ymax"))



    basename = os.path.basename(path)

    basename = os.path.splitext(basename)[0]

    class_name = basename[:basename.rfind("_")].lower()

    if class_name not in class_names:

        class_names[class_name] = k

        k += 1



    output.append((path, height, width, xmin, ymin, xmax, ymax, class_name, class_names[class_name]))



# preserve percentage of samples for each class ("stratified")

output.sort(key=lambda tup : tup[-1])
lengths = []

i = 0

last = 0

for j, row in enumerate(output):

    if last == row[-1]:

        i += 1

    else:

        print("class {}: {} images".format(output[j-1][-2], i))

        lengths.append(i)

        i = 1

        last += 1



print("class {}: {} images".format(output[j-1][-2], i))

lengths.append(i)
training_data = []

validation_data = []

s = 0

for c in lengths:

    for i in range(c):

        path, height, width, xmin, ymin, xmax, ymax, class_name, class_id = output[s]

        if xmin >= xmax or ymin >= ymax or xmax > width or ymax > height or xmin < 0 or ymin < 0:

            print("Warning: {} contains invalid box. Skipped...".format(path))

            continue

            

        if i <= c * SPLIT_RATIO:

            training_data.append(output[s])

        else:

            validation_data.append(output[s])



        s += 1

print(len(training_data))

print(len(validation_data))