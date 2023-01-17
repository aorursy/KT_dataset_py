import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json
import seaborn as sns
from datetime import datetime
import os
%matplotlib inline

# loading datasets as json
with open("../input/iwildcam-2020-fgvc7/iwildcam2020_megadetector_results.json") as json_file:
    megadetector_results = json.load(json_file)
with open("../input/iwildcam-2020-fgvc7/iwildcam2020_test_information.json") as json_file:
    test_information = json.load(json_file)
with open("../input/iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json") as json_file:
    train_annotations = json.load(json_file)

# converting into dataframes
megadetector_images = pd.DataFrame(megadetector_results["images"])
megadetector_cat = megadetector_results["detection_categories"] # not a dataframe

test_cat = pd.DataFrame(test_information["categories"])
test_images = pd.DataFrame(test_information["images"])

train_annot = pd.DataFrame(train_annotations["annotations"])
train_images = pd.DataFrame(train_annotations["images"])
train_cat = pd.DataFrame(train_annotations["categories"])

sample_submission = pd.read_csv("/kaggle/input/iwildcam-2020-fgvc7/sample_submission.csv")

#images
train_jpg = glob.glob('../input/iwildcam-2020-fgvc7/train/*')
test_jpg = glob.glob('../input/iwildcam-2020-fgvc7/test/*')

# # some files are giving error later. So its better to get rid of them right now. This takes a lot of time 
# to_remove = []
# count = 0
# for i in train_jpg:
#     count+=1
#     try:
#         img = Image.open(i)
#     except:
#         to_remove.append(i)
#     if (count%100) == 0:
#         print(count)

# to_remove

# for i in to_remove:
#     train_jpg.remove(i)

# these are the files
for i in ['../input/iwildcam-2020-fgvc7/train/87022118-21bc-11ea-a13a-137349068a90.jpg',
 '../input/iwildcam-2020-fgvc7/train/8f17b296-21bc-11ea-a13a-137349068a90.jpg',
 '../input/iwildcam-2020-fgvc7/train/8792549a-21bc-11ea-a13a-137349068a90.jpg',
 '../input/iwildcam-2020-fgvc7/train/883572ba-21bc-11ea-a13a-137349068a90.jpg',
 '../input/iwildcam-2020-fgvc7/train/896c1198-21bc-11ea-a13a-137349068a90.jpg',
 '../input/iwildcam-2020-fgvc7/train/99136aa6-21bc-11ea-a13a-137349068a90.jpg']:
    train_jpg.remove(i)
    train_images.drop(train_images[train_images.file_name == i[35:]].index ,axis = 0, inplace= True)

# megadetector
def special_func(list_x):
    '''
    Will be used below to make sure all bounding boxes are entered when there are multiple animals detected in an image. 
    '''
    list_return = []
    for i in list_x:
        list_return.append(i["bbox"])
    return list_return

megadetector_images["conf"] = megadetector_images.detections.apply(lambda x: float(x[0]["conf"]) if x!=[] else 0)
megadetector_images["category"] = megadetector_images.detections.apply(lambda x: x[0]["category"] if x!=[] else 0)
megadetector_images["bbox"] = megadetector_images.detections.apply(lambda x: [x[0]["bbox"]] if len(x)==1 else (special_func(x) if len(x)>1 else []))
megadetector_images["cat"] = megadetector_images.category.map(megadetector_cat)
megadetector_images.drop(["max_detection_conf", "detections"], axis = 1, inplace= True)
megadetector_images["category"] = megadetector_images.category.apply(lambda x: int(x))

mega = megadetector_images.copy()

# making final test
temp_test = pd.merge(test_images, megadetector_images, how = "left", on = "id")

# making a unified train dataset
train_images.rename({"id":"image_id"}, axis = 1, inplace= True)

train = train_images.merge(train_annot, on = "image_id", how = "inner")

train = train.merge(train_cat.drop("count", axis=1).rename({"id":"category_id"}, axis = 1))

# making main df
df = train.merge(mega.rename({"id":"image_id"}, axis = 1), how = "left")

# handling null entries
df.conf = df.conf.fillna(0.0)
df.category = df.category.fillna(0)
df.loc[df.bbox[df.bbox.isnull()].index,"bbox"] = [[[]] * df.bbox.isnull().sum()]
df.loc[df.cat[df.cat.isnull()].index, "cat"] = "none"

temp_test.conf = temp_test.conf.fillna(0.0)
temp_test.category = temp_test.category.fillna(0)
temp_test.loc[temp_test.bbox[temp_test.bbox.isnull()].index,"bbox"] = [[[]] * temp_test.bbox.isnull().sum()]
temp_test.loc[temp_test.cat[temp_test.cat.isnull()].index, "cat"] = "none"

# making sure the top 2 bboxes for each image have been included as different rows
bbox_df = pd.DataFrame.from_records(df.bbox)
bbox_df = bbox_df.loc[:,:1]

empty_df = pd.DataFrame(columns = df.columns)
df = pd.concat([df, bbox_df], axis = 1)

cols = bbox_df.columns
for i in cols:
    remove_list = [x for x in cols if x != i] 
    remove_list.append("bbox")
    current = df.drop(remove_list, axis = 1)
    current.rename({i:"bbox"}, axis = 1, inplace= True)
    current.dropna(inplace = True)
    empty_df = pd.concat([empty_df, current], axis = 0)

# concatenating the records with empty bbox
df = pd.concat([empty_df, df.iloc[df.bbox.index[df.bbox.apply(lambda x: x == []) == True],:-2]], axis = 0)

print(df.shape)
print(temp_test.shape)
temp_test.to_pickle("test.pkl")
df.to_pickle("df.pkl")
### megadetector
# # checking if conf and max_detection_conf are the same or not
# sum(megadetector_images.conf == megadetector_images.max_detection_conf) == megadetector_images.shape[0]

# print(megadetector_images.category.unique())
# print(megadetector_images.cat.unique())
# print(megadetector_cat)

# def check_megadetector_img(img_id):
#     complete_id = '../input/iwildcam-2020-fgvc7/train/'+img_id+'.jpg'
#     plt.figure(figsize = (6,5))
#     img = Image.open(complete_id)
#     plt.imshow(img)
#     print(megadetector_images[megadetector_images.id ==img_id][["cat", "conf", "bbox"]])

# # check images and respective confidence, category and bbox 
# check_megadetector_img(megadetector_images.loc[4,"id"])

# # distribution of categories in megadetector
# sns.countplot(megadetector_images.fillna("missing").cat)

### train
# confirming that id column has unique ids
# print(train_annot.image_id.value_counts()[train_annot.image_id.value_counts()>1])
# print(train_images.id.value_counts()[train_images.id.value_counts()>1])
# plt.figure(figsize = (50,10))
# sns.countplot(train.category_id)
# plt.title("Distirbution of Categories")
from datetime import datetime
import cv2

sub = df.head(100)

df.datetime = pd.to_datetime(df.datetime)

df["time"] = df.datetime.apply(lambda x: x.time())
sub = df[df.time > datetime.strptime("23:00:00", "%H:%M:%S").time()]
im = Image.open("../input/iwildcam-2020-fgvc7/train/" + sub["file_name"].iloc[0])
plt.imshow(im)
sub.iloc[0].bbox
sub.iloc[62].bbox[1] - sub.iloc[62].bbox[0]
im = cv2.imread("../input/iwildcam-2020-fgvc7/train/" + sub["file_name"].iloc[62])
plt.imshow(im)
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)
im_gamma = adjust_gamma(im, 1.5)
plt.imshow(im_gamma)
np.mean(im)