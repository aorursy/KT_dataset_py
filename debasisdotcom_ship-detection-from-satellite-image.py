import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import cv2

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from tqdm import tqdm_notebook, tnrange

from tqdm import tqdm
main_ship_data = pd.read_csv("../input/airbus-ship-detection/train_ship_segmentations_v2.csv")
main_ship_data["has_ship"] = main_ship_data["EncodedPixels"].map(lambda x:1 if isinstance(x,str) else 0)
main_ship_data.head()
unique_ship_data = main_ship_data.groupby("ImageId").agg({"has_ship":sum}).reset_index()

unique_ship_data["number_of_ships"] = unique_ship_data["has_ship"]

unique_ship_data.drop("has_ship", axis = 1, inplace = True)



unique_ship_data["has_ship"] = unique_ship_data["number_of_ships"].map(lambda x:1.0 if x>=1.0 else 0)
unique_ship_data.head()
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

    

   xC = (x1+x0)/(2*768)

   yC = (y1+y0)/(2*768)

   h = np.abs(y1-y0)/768

   w = np.abs(x1-x0)/768



   return [xC, yC, h, w]
main_ship_data["bbox"] = main_ship_data["EncodedPixels"].map(lambda x: rle2bbox(x, (768,768)) if isinstance(x,str) else np.NaN )

main_ship_data.drop("EncodedPixels", axis=1, inplace = True)
#Creating a new column with the area of bounding box in it.

main_ship_data["bboxArea"]=main_ship_data["bbox"].map(lambda x:x[2]*768*x[3]*768 if x==x else 0)
main_ship_data.head()
# Plotting the distribution of the bounding box areas to check the ship sizes



area = main_ship_data[main_ship_data["has_ship"]>0]



plt.figure(figsize = (12,5))

plt.subplot(1,2,1)

sns.boxplot(area["bboxArea"])

plt.title("Areas of Bounding boxes for ships")

# plt.xscale("log")

plt.subplot(1,2,2)

plt.hist(area["bboxArea"], bins=50)

# plt.xscale("log")

plt.title("Distribution of Bounding box area")

plt.xlabel("Bounding Box Area")

plt.tight_layout()
area = main_ship_data[(main_ship_data["has_ship"]>0)&(main_ship_data["bboxArea"]<20)]

area["bboxArea"] = np.round(area["bboxArea"])



plt.figure(figsize=(10,5))

sns.countplot(x="bboxArea", data=area)

plt.xlabel("Area of bounding box")

plt.ylabel("Number of Images")

plt.show()
# Finding the distribution of no of ships



plt.figure(figsize = (20,10))

plt.subplot(2,2,1)

classes=["No Ship","Ship"]

ax = sns.countplot(unique_ship_data["has_ship"])

ax.set_xticklabels(classes)

plt.ylabel("Number of Images")

plt.title("Images with ship vs Without Ship")



plt.subplot(2,2,2)

sns.countplot(unique_ship_data["number_of_ships"])

# plt.yscale("log")

plt.xlabel("Number of Ships")

plt.ylabel("Number of Images")

plt.title("Number of Ships count (Including no ship)")



withship = unique_ship_data[unique_ship_data["has_ship"]==1]

plt.subplot(2,2,3)

sns.countplot(withship["number_of_ships"])

plt.xlabel("Number of Ships")

plt.ylabel("Number of Images")

plt.title("Number of Ships count (Excluding no ship)")



plt.subplot(2,2,4)

sns.boxplot(withship["number_of_ships"])

plt.xlabel("Number of Ships")

plt.title("Distribution of number of ships(Excluding no ship)")



plt.tight_layout()
# Removing boxes which are less than 1 percentile

# main_ship_data = main_ship_data[(main_ship_data["bboxArea"]>10) & (main_ship_data["has_ship"]==1)]



main_ship_data = main_ship_data.drop(main_ship_data[(main_ship_data["bboxArea"]<10) & (main_ship_data["has_ship"]!=0)].index)
numberofships = 1000



balance_ship_data = unique_ship_data.groupby("number_of_ships").apply(lambda x:x.sample(numberofships) if len(x)>numberofships else x)

balance_ship_data = balance_ship_data.reset_index(drop = True)

balance_ship_data = balance_ship_data.drop("has_ship", axis=1)
balance_ship_data
balance_ship_data["number_of_ships"].hist(bins=16)

plt.xlabel("Number of Ships")

plt.ylabel("Number of Images")

plt.title("Number of Ships count on unique Image_Id ")

plt.show()
# Merging the the balance_ship_data with main_ship_data in order to have the Encodedpixels

balance_ship_data = pd.merge(balance_ship_data, main_ship_data, on='ImageId')
balance_ship_data.sample(5)
#Distrubution of count of images with ships and no ships in balance_ship_data dataframe

classes=["No Ship","Ship"]

sns.set_style('darkgrid')  

ax = sns.countplot(x="has_ship", data=balance_ship_data)

ax.set_xticklabels(classes)

plt.ylabel("Number of Images")

plt.show()
#Distrubution of number of ships in a single image in balance_ship_data dataframe

sns.set_style('darkgrid')  

ax = sns.countplot(x="number_of_ships", data=balance_ship_data)

# ax.set_xticklabels(classes)

plt.xlabel("Number of Ships")

plt.ylabel("Number of Images")

plt.show()
#Defining a function to load image

def load_img(path):

    image = cv2.imread(path)

    return image[...,::-1]
path = "../input/airbus-ship-detection/train_v2/"



plt.figure(figsize=(20, 15))



for i in range(16):

    

  imageid = balance_ship_data["ImageId"][balance_ship_data["number_of_ships"]==i].sample(5).reset_index(drop=True)    

  imageid = imageid[0]

  image = np.array(load_img(path+imageid))



  text = "Name of the image:{0}".format(imageid[0])

    

  Bbox = balance_ship_data["bbox"][balance_ship_data["ImageId"]==imageid].reset_index(drop=True)



  plt.subplot(4,5,i+1)

    

  if i>0:

    for j in Bbox:

      # print(i[0])

      xc = j[0]

      yc = j[1]

      h = j[2]

      w = j[3]



      x0 = int((xc-(w/2))*768)

      y0 = int((yc-(h/2))*768)

      x1 = int((xc+(w/2))*768)

      y1 = int((yc+(h/2))*768)



      cv2.rectangle(image,

            pt1=(x0,y0),

            pt2=(x1,y1),

            color=(255,0,0),

            thickness=3)

    

  plt.imshow(image)

  plt.title("Number of ship:{}".format(i),fontsize=10)

  plt.axis('off')

  

plt.tight_layout()

plt.show()
# folder_location = "/content/drive/My Drive/Colab_Notebooks/Capstone" ##Location of the folder which contains the train and test images



# for i, img_id in tqdm(enumerate(balance_ship_data["ImageId"])):



#     filt_df = balance_ship_data[balance_ship_data.ImageId==img_id]

#     all_boxes = filt_df.bbox.values

#     img_id = img_id.split(".")[0]

#     file_name = "{}/{}.txt".format(folder_location,img_id) 



#     s = "0 %s %s %s %s \n" 

#     with open(file_name, 'a') as file: 

#         if filt_df["has_ship"]>0:

#             for i in all_boxes:

#                 new_line = (s % tuple(i))

#                 file.write(new_line)
# X = balance_ship_data[["ImageId"]]

# y = balance_ship_data["EncodedPixels"]



# train, test, _, _ = train_test_split(X, y, test_size = 0.2, random_state = 2)
# path = "/content/Capstone/shipimages/" #path where ship images are there

# path_txt = "/content/Capstone/"        #path where you want the txt file to be created



# train["ImageId"] = path + train["ImageId"]

# test["ImageId"] = path +test["ImageId"]
# train.to_csv(path_txt+"Train_path.txt",index=None, header=None, sep=" ", mode="a")

# test.to_csv(path_txt+"Test_path.txt",index=None, header=None, sep=" ", mode="a")