# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm_notebook as tqdm
import shutil
ships = pd.read_csv("/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv")
test_data = pd.read_csv("/kaggle/input/airbus-ship-detection/sample_submission_v2.csv")
ships.head()
ships["Ship"] = ships["EncodedPixels"].map(lambda x:1 if isinstance(x,str) else 0)
ship_unique = ships[["ImageId","Ship"]].groupby("ImageId").agg({"Ship":"sum"}).reset_index()
ship_unique.head()
def rle2bbox(rle, shape):
    
    a = np.fromiter(rle.split(), dtype=np.uint)
    a = a.reshape((-1, 2))  # an array of (start, length) pairs
    a[:,0] -= 1  # `start` is 1-indexed
    
    y0 = a[:,0] % shape[0]
    y1 = y0 + a[:,1]
    if np.any(y1 > shape[0]):
        # got `y` overrun, meaning that there are a pixels in mask on 0 and shape[0] position
        y0 = 0
        y1 = shape[0]
    else:
        y0 = np.min(y0)
        y1 = np.max(y1)
    
    x0 = a[:,0] // shape[0]
    x1 = (a[:,0] + a[:,1]) // shape[0]
    x0 = np.min(x0)
    x1 = np.max(x1)
    
    if x1 > shape[1]:
        # just went out of the image dimensions
        raise ValueError("invalid RLE or image dimensions: x1=%d > shape[1]=%d" % (
            x1, shape[1]
        ))

    xc = (x0+x1)/(2*768)
    yc = (y0+y1)/(2*768)
    w = np.abs(x1-x0)/768
    h = np.abs(y1-y0)/768
    return [xc, yc, h, w]
# Finding the Bounding boxes from encoded pixels Normalized
ships["Bbox"] = ships["EncodedPixels"].apply(lambda x:rle2bbox(x,(768,768)) if isinstance(x,str) else np.NaN)
ships.head()
# Droping the Encoded Pixels from the dataset
ships.drop("EncodedPixels", axis =1, inplace =True)
ships["BboxArea"]=ships["Bbox"].map(lambda x:x[2]*768*x[3]*768 if x==x else 0)
# Plotting the distribution of the bounding box areas to check the ship sizes

area = ships[ships.Ship>0]

plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
sns.boxplot(area["BboxArea"])
plt.title("Areas of Bounding boxes for ships")
plt.xscale("log")
plt.subplot(1,2,2)
sns.distplot(area["BboxArea"], bins=50)
plt.xscale("log")
plt.title("Distribution of Bounding boxe area")
plt.tight_layout()

np.percentile(area["BboxArea"],[1,5,25,50,75,95,99])
# Removing boxes which are less than 1 percentile
ships = ships[ships["BboxArea"]>np.percentile(ships["BboxArea"],1)]
# Finding the distribution of no of ships
ship_unique["Hasship"]= [1 if x>0 else 0 for x in ship_unique["Ship"]]
plt.figure(figsize = (12,12))
plt.subplot(2,2,1)
sns.countplot(ship_unique["Hasship"])
plt.title("Images with ship vs Without Ship")
plt.subplot(2,2,2)
sns.countplot(ship_unique["Ship"])
plt.title("No of Ships count")
withship = ship_unique[ship_unique["Hasship"]==1]
plt.subplot(2,2,3)
sns.countplot(withship["Ship"])
plt.title("Count of no of ships(Excluding no ship)")
plt.subplot(2,2,4)
sns.boxplot(withship["Ship"])
plt.title("Distribution of no of ships(Excluding no ship)")
plt.tight_layout()
# Since the data set volumn is very high, we are downsampling to select 1000 images from each of the classes(Where more than 1000)
balanced_df = ship_unique.groupby("Ship").apply(lambda x:x.sample(1000) if len(x)>=1000 else x.sample(len(x)))
balanced_df.reset_index(drop=True,inplace=True)
balanced_df["Ship"].hist(bins=16)
print(balanced_df.sample(5))
print(balanced_df.shape)
# Creating dataframe for Bounding boxes for the images in Balanced_df
balanced_bbox = ships.merge(balanced_df[["ImageId"]], how ="inner", on = "ImageId")
balanced_bbox.head()
# Visualizing the bounding boxes and images
path ="../input/airbus-ship-detection/train_v2/"
plt.figure(figsize =(20,20))
for i in range(15):
    imageid = balanced_df[balanced_df.Ship ==i].iloc[0][0]
    image = np.array(cv2.imread(path+imageid)[:,:,::-1])
    if i>0:
        bbox = balanced_bbox[balanced_bbox.ImageId==imageid]["Bbox"]
        
        for items in bbox:
            Xmin  = int((items[0]-items[3]/2)*768)
            Ymin  = int((items[1]-items[2]/2)*768)
            Xmax  = int((items[0]+items[3]/2)*768)
            Ymax  = int((items[1]+items[2]/2)*768)
            cv2.rectangle(image,
                          (Xmin,Ymin),
                          (Xmax,Ymax),
                          (255,0,0),
                          thickness = 2)
    plt.subplot(4,4,i+1)
    plt.imshow(image)
    plt.title("No of ships = {}".format(i))

# Creating New folder to place the image and Text files
import os
if not os.path.exists("train_data_yolo"):
    os.makedirs("train_data_yolo")
# Creatint Text Files for the images in balanced dataframe
folder_location = "train_data_yolo"
for img_id in tqdm(balanced_df["ImageId"]):
    
#     k+1# loop through all unique image ids. Remove the slice to do all images
    bbox = ships[["ImageId","Ship","Bbox"]][ships.ImageId==img_id]
    all_boxes = bbox.Bbox.values
    img_id = img_id.split(".")[0]
    file_name = "{}/{}.txt".format(folder_location,img_id) 
    s = "0 %s %s %s %s \n" 
    with open(file_name, 'a') as file: # append lines to file
        if bbox.Ship.iloc[0]>0:
            for i in all_boxes:
                new_line = (s % tuple(i))
                file.write(new_line)
# Transfering the images to the newly created folder where text files were created
directory = "../input/airbus-ship-detection/train_v2/" 
for img_id in tqdm(balanced_df["ImageId"]):
    image_id = directory+img_id
    shutil.copy(image_id, "train_data_yolo")

