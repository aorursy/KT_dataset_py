# basic imports
import os, random
import pandas as pd
import numpy as np
# what do we have here?
print (os.listdir("../input"))
TRAIN_LABELS_CSV_FILE="../input/stage_1_train_labels.csv"
# pedantic nit: we are changing 'Target' to 'label' on the way in
TRAIN_LABELS_CSV_COLUMN_NAMES=['patientId', 'x1', 'y1', 'width', 'height', 'label']

# we will pre-process bounding boxes into the following format
# we will add x2=x1+width and y2=x2+height
# NaN rows (non 'Lung Opacity' rows) which do not have bounding boxes will be discarded
TRAIN_BOUNDINGBOX_CSV_FILE="stage_1_train_boundingboxes.csv"
TRAIN_BOUNDINGBOX_CSV_COLUMN_NAMES=['patientId', 'x1', 'y1', 'width', 'height', 'x2', 'y2']

# we will compute 'superset' bounding boxes for each patientId
TRAIN_COMBINED_BOUNDINGBOX_CSV_FILE="stage_1_train_combinedboxes.csv"
TRAIN_COMBINED_BOUNDINGBOX_CSV_COLUMN_NAMES=['patientId', 'x1min', 'y1min', 'maxwidth', 'maxheight', 'x2max', 'y2max']
# read TRAIN_LABELS_CSV_FILE into a pandas dataframe
labelsbboxdf = pd.read_csv(TRAIN_LABELS_CSV_FILE,
                           names=TRAIN_LABELS_CSV_COLUMN_NAMES,
                           # skip the header line
                           header=0,
                           # index the dataframe on patientId
                           index_col='patientId')
print (labelsbboxdf.shape)
#print (labelsbboxdf.head(n=10))
# grab labels by unique patienId
labelsdf=pd.DataFrame(labelsbboxdf.pop('label'), columns=['label'])
# remove duplicates
labelsdf=pd.DataFrame(labelsdf.groupby(['patientId'])['label'].first(), columns=['label'])
print (labelsdf.shape)
#print (labelsdf.head(n=10))
# after 'label' is popped off, x1,y1,width,height are left in labelsbboxdf
# drop missing values (all rows except ones labeled as having  'Lung Opacity' will be dropped)
bboxesdf=labelsbboxdf.dropna()
print (bboxesdf.shape)
#print(bboxesdf.head(n=10))

# create coordinates for right hand bottom corner for all bounding boxes
bboxesdf['x2']=bboxesdf['x1']+bboxesdf['width']
bboxesdf['y2']=bboxesdf['y1']+bboxesdf['height']
print (bboxesdf.shape)
#print(bboxesdf.head(n=10))
# let's view the bounding box information with the new fields
bboxesdf.head(n=10)
# what can we glean from the raw bounding boxes?
bboxesdf.describe(percentiles=[0.05, 0.95])
# the bounding boxes are situated across the entire image dimensions
# from lowest values of x1,y1=2.0,2.0 to largest values of x2,y2=1024,1024

# we can focus our classification and segmentation efforts inside the top-left
# and bottom-right corners created at 5th and 95th percentile boundaries by
# (x1, y1) and (x2, y2) respectively

# create unique bounding boxes by patientId that 'subsume' smallest and largest endpoints
x1y1df=bboxesdf.groupby(['patientId'])['x1', 'y1'].apply(np.min, axis=0)
x2y2df=bboxesdf.groupby(['patientId'])['x2', 'y2'].apply(np.max, axis=0)
combinedbboxdf=pd.concat([x1y1df, x2y2df], axis=1)
# modify the column names to provide context
combinedbboxdf.rename(columns={'x1':'x1min', 'y1':'y1min', 'x2':'x2max', 'y2':'y2max'}, inplace=True)
# recompute width and height for combined bounding boxes
combinedbboxdf['maxwidth']=combinedbboxdf['x2max']-combinedbboxdf['x1min']
combinedbboxdf['maxheight']=combinedbboxdf['y2max']-combinedbboxdf['y1min']
# make the column order consistent
combinedbboxdf=combinedbboxdf[['x1min', 'y1min', 'maxwidth', 'maxheight', 'x2max', 'y2max']]
# let's view the superset bounding boxes
combinedbboxdf.head(n=10)
# what can we glean from the superset bounding boxes?
combinedbboxdf.describe(percentiles=[0.05, 0.95])
# again, the bounding boxes are situated across the entire image dimensions,
# but we can focus inside the 5th and 95th percentile boundaries as above
bboxesdf.to_csv(TRAIN_BOUNDINGBOX_CSV_FILE)
combinedbboxdf.to_csv(TRAIN_COMBINED_BOUNDINGBOX_CSV_FILE)
!head -10 stage_1_train_boundingboxes.csv
!head -10 stage_1_train_combinedboxes.csv