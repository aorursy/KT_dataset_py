import pandas as pd
from scipy.io import loadmat 
from pathlib import Path

datapath = Path('../input/')
MAT = loadmat(datapath/'cars_annos.mat')

print("Annotations")
print(MAT["annotations"][0,:5])
print("Class Names")
print(MAT["class_names"][0][:5])
def get_labels():
    MAT = loadmat(datapath/'cars_annos.mat')
    annotations = MAT["annotations"][0,:]
    nclasses = len(MAT["class_names"][0])
    class_names = dict(zip(range(1,nclasses),[c[0] for c in MAT["class_names"][0]]))
    
    labelled_images = {}
    dataset = []
    for arr in annotations:
        # the first entry in the row is the image name
        # The rest is the data, first bbox, then classid then a boolean for whether in train or test set
        dataset.append([arr[0][0].replace('car_ims/','')] + [y[0][0] for y in arr][1:])
    # Convert to a DataFrame, and specify the column names
    DF = pd.DataFrame(dataset, 
                      columns =['filename',"BBOX_Y2","BBOX_X1","BBOX_Y1","BBOX_X2","ClassID","TestSet"])

    DF = DF.assign(ClassName=DF.ClassID.map(dict(class_names)))
    return DF

DF = get_labels()
DF.head()
from pylab import imread,subplot,imshow,show
import matplotlib.pyplot as plt

image = imread(datapath/'cars_train/cars_train/00001.jpg')  
plt.imshow(image)
image = imread(datapath/'cars_test/cars_test/00001.jpg')  #// choose image location

plt.imshow(image)