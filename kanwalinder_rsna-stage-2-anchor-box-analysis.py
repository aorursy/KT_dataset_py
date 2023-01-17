import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
# clone darknet
!git clone https://github.com/pjreddie/darknet
# lets look at the default anchor boxes in yolov3-tiny.cfg (6 anchor boxes)
# and yolov3.cfg (9 anchor boxes) and the associated input image sizes
!cp darknet/cfg/yolov3-tiny.cfg .
!grep -E 'width|height|anchors' yolov3-tiny.cfg

print ("---------")

!cp darknet/cfg/yolov3.cfg .
!grep -E 'width|height|anchors' yolov3.cfg
# so the default configuration of yolov3-tiny is 416x416, and the one for yolov3 is 608x608
# yolov3 uses three anchor boxes per 'scale.'  Predictions are first done at the input scale.
# then the network upsamples the inputs to twice the resolution and makes predictions again.
# upsampling helps detect smaller objects.  yolov3-tiny has 2 scales (6 anchor boxes), and
# yolov3 has 3 scales (9 anchor boxes).
# Analysis Choices:
# V2: input width and height of 608 (19x19 cells) for both YOLOV3 Tiny and YOLOV3
# V3: input width and height of 512 (16x16 cells) for both YOLOV3 Tiny and YOLOV3
# cleanup darknet download
!rm -rf darknet
# global variables
TRAIN_LABELS_CSV_FILE="../input/stage_2_train_labels.csv"
# pedantic nit: we are changing 'Target' to 'label' on the way in
TRAIN_LABELS_CSV_COLUMN_NAMES=['patientId', 'x1', 'y1', 'bw', 'bh', 'label']

DICOM_IMAGE_SIZE=1024
YOLOV3_SIZE=512
# read RSNA TRAIN_LABELS_CSV_FILE into a pandas dataframe
labelsbboxdf = pd.read_csv(TRAIN_LABELS_CSV_FILE,
                           names=TRAIN_LABELS_CSV_COLUMN_NAMES,
                           # skip the header line
                           header=0,
                           # index the dataframe on patientId
                           index_col='patientId')

labelsbboxdf.head(10)
# drop all fields except the bounding box dimensions and
# all row except the Lung Opacity ones
yolov3bboxesdf=labelsbboxdf[['bw', 'bh']].dropna()
yolov3bboxesdf.head(10)
# resize bounding boxes for YOLOV3_SIZE
yolov3bboxesdf=yolov3bboxesdf*(YOLOV3_SIZE/DICOM_IMAGE_SIZE)
yolov3bboxesdf.head(10)
# as reference, below are the vitals on bounding boxes at DICOM_IMAGE_SIZE
labelsbboxdf[['bw', 'bh']].describe(percentiles=[0.25, 0.5, 0.75, .95])
# below are the vitals on bounding boxes at chosen input size of YOLOV3_SIZE
yolov3bboxesdf.describe(percentiles=[0.25, 0.5, 0.75, 0.85, .95])
# we could hand-craft the following anchor boxes :
# ~<min, ~<25%, ~<50%, ~<75%, ~<85%, ~<95% and have a 6 anchor box set (for yolov3 tiny)
!printf '10,15, 75,75 100,125, 100,175 125,225, 150,275\n' > rsna-yolov3-manual-tiny-anchors.txt
!cat rsna-yolov3-manual-tiny-anchors.txt
# let's see what kmeans analysis gives us
# convert to numpy array
bboxarray=np.array(yolov3bboxesdf)

print (bboxarray.shape)
print (bboxarray)
# fit to 6 kmeans clusters (for yolov3 tiny)
kmeans=MiniBatchKMeans(n_clusters=6, verbose=1)
colors=['b.', 'g.', 'r.', 'c.', 'm.', 'y.',  'k.']
kmeans.fit(bboxarray)
centroids=kmeans.cluster_centers_
labels=kmeans.labels_

print (centroids.shape)
print (centroids)
print (labels.shape)
print (labels)
# view computed centroids to bounding box dimensions' scatterplot
plt.figure(figsize=(10,10))
for i in range(len(bboxarray)):
    plt.plot(bboxarray[i][0], bboxarray[i][1], colors[labels[i]], markersize=10)   
plt.scatter(centroids[:,0], centroids[:,1], marker="x", s=150, linewidth=5, zorder=10)
plt.show()
# post process centroids
anchors=np.around(centroids)
print (len(anchors))
print (anchors)
print ("---------")
ind = np.lexsort((anchors[:,1], anchors[:,0])) # lexsort uses the second argument first, followed by the first argument
#print (ind)
sortedanchors=np.array([anchors[i] for i in ind])
print(sortedanchors)
# write anchor boxes to file
# organize anchor boxes in YOLOV3 format
for i in range (len(sortedanchors)):
    anchorbox="{},{}".format(int(sortedanchors[i][0]), int(sortedanchors[i][1]))
    if i==0:
        anchorrecord=anchorbox
    else:
        anchorrecord="{},  {}".format(anchorrecord, anchorbox)
anchorrecord="{}\n".format(anchorrecord)

print (anchorrecord)

# save anchor box specification to file
savedanchorsfilename='rsna-yolov3-kmeans-tiny-anchors.txt'
with open(savedanchorsfilename,'w') as file:
    file.write(anchorrecord)
file.close()
!cat rsna-yolov3-kmeans-tiny-anchors.txt
# fit to 9 kmeans clusters
kmeans=MiniBatchKMeans(n_clusters=9, verbose=1)
colors=['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'w.', 'b.']
kmeans.fit(bboxarray)
centroids=kmeans.cluster_centers_
labels=kmeans.labels_

print (centroids.shape)
print (centroids)
print (labels.shape)
print (labels)
# view computed centroids to bounding box dimensions' scatterplot
plt.figure(figsize=(10,10))
for i in range(len(bboxarray)):
    plt.plot(bboxarray[i][0], bboxarray[i][1], colors[labels[i]], markersize=10)   
plt.scatter(centroids[:,0], centroids[:,1], marker="x", s=150, linewidth=5, zorder=10)
plt.show()
# post process centroids
anchors=np.around(centroids)
print (len(anchors))
print (anchors)
print ("---------")
ind = np.lexsort((anchors[:,1], anchors[:,0]))
#print (ind)
sortedanchors=np.array([anchors[i] for i in ind])
print(sortedanchors)
# write anchor boxes to file
# organize anchor boxes in YOLOV3 format
for i in range (len(sortedanchors)):
    anchorbox="{},{}".format(int(sortedanchors[i][0]), int(sortedanchors[i][1]))
    if i==0:
        anchorrecord=anchorbox
    else:
        anchorrecord="{},  {}".format(anchorrecord, anchorbox)
anchorrecord="{}\n".format(anchorrecord)

print (anchorrecord)

# save anchor box specification to file
savedanchorsfilename='rsna-yolov3-kmeans-anchors.txt'
with open(savedanchorsfilename,'w') as file:
    file.write(anchorrecord)
file.close()
!cat rsna-yolov3-kmeans-anchors.txt
# everything together
# print default yolov3-tiny anchors
!grep anchors yolov3-tiny.cfg
# print hand-crafted yolov3 tiny anchors
!cat rsna-yolov3-manual-tiny-anchors.txt
# print kmeans suggested yolov3 tiny anchors
!cat rsna-yolov3-kmeans-tiny-anchors.txt
# print default yolov3 anchors
!grep anchors yolov3.cfg
# print kmeans suggested yolov3 anchors
!cat rsna-yolov3-kmeans-anchors.txt
