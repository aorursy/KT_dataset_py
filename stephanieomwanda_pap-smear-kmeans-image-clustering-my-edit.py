# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!ls "/kaggle/input/"
!ls ../input/
%matplotlib inline

from fastai import *

from fastai.vision import *

import pandas as pd

import matplotlib.image as mpimg
#Path to where the data is located

data_path = Path("/kaggle/input/pap-smear-datasets/")



#path to the herlev pap smear dataset

herlev_path = data_path/"herlev_pap_smear"



#path to the herlevr sipakmed fci pap smear dataset

sipakmed_fci_path = data_path/"sipakmed_fci_pap_smear"



#path to the herlevr sipakmed wsi pap smear dataset

sipakmed_wsi_path = data_path/"sipakmed_wsi_pap_smear"
herlev_path.ls()
class_paths = herlev_path.ls()

for c in class_paths:

    img_in_c_paths = get_image_files(c)

    print(f"Number of images in '{c.name}' : {len(img_in_c_paths)}")
sipakmed_fci_path.ls()
class_paths = sipakmed_fci_path.ls()

for c in class_paths:

    img_in_c_paths = get_image_files(c)

    print(f"Number of images in '{c.name}' : {len(img_in_c_paths)}")
sipakmed_wsi_path.ls()
class_paths = sipakmed_wsi_path.ls()

for c in class_paths:

    img_in_c_paths = get_image_files(c)

    print(f"Number of images in '{c.name}' : {len(img_in_c_paths)}")
tfms = get_transforms(flip_vert=True, max_warp=0.0, max_zoom=0.)

herlev_data_block = (ImageList.from_folder(herlev_path)

                    .filter_by_func(lambda fname: "-d" not in fname.name)

                    .split_by_rand_pct(valid_pct=0.2, seed=0)

                    .label_from_func(lambda fname: "abnormal" if "abnormal" in fname.parent.name else "normal")

                    .transform(tfms, size=128)

                    .databunch(bs=16)

                    .normalize(imagenet_stats))
herlev_data_block
herlev_data_block.show_batch(rows=4, figsize=(10 ,10))
def labelling_func(fname):

    c = fname.parent.name

    if "abnormal" in c:

        return "abnormal"

    elif "benign" in c:

        return "abnormal"

    else:

        return "normal"



tfms = get_transforms(flip_vert=True, max_warp=0.0, max_zoom=0.9)



sipakmed_fci_data_block = (ImageList.from_folder(sipakmed_fci_path)

                      .split_by_rand_pct(valid_pct=0.2, seed=42)

                      .label_from_func(labelling_func)

                      .transform(tfms, size=128)

                      .databunch(bs=16)

                      .normalize(imagenet_stats))
sipakmed_fci_data_block
sipakmed_fci_data_block.show_batch(rows=4, figsize=(10 ,10))
def labelling_func(fname):

    c = fname.parent.name

    if "abnormal" in c:

        return "abnormal"

    elif "benign" in c:

        return "abnormal"

    else:

        return "normal"



tfms = get_transforms(flip_vert=True, max_warp=0.0, max_zoom=0.9)



sipakmed_wsi_data_block = (ImageList.from_folder(sipakmed_wsi_path)

                      .split_by_rand_pct(valid_pct=0.2, seed=42)

                      .label_from_func(labelling_func)

                      .transform(tfms, size=128)

                      .databunch(bs=16)

                      .normalize(imagenet_stats))
sipakmed_wsi_data_block
sipakmed_wsi_data_block.show_batch(rows=4, figsize=(10 ,10))
classes = sipakmed_fci_path.ls()

classes[0].ls()
sorted_skipamed_files = sorted(classes[0].ls())

sorted_skipamed_files
sample_image_and_coordinates = sorted_skipamed_files[:3]

sample_image_and_coordinates
sample_img = sample_image_and_coordinates[0]

open_image(sample_img)
nucleus_data = pd.read_csv(sample_image_and_coordinates[1], header=None)

nucleus_data.head()
cytoplasm_data = pd.read_csv(sample_image_and_coordinates[2], header=None)

cytoplasm_data.head()
img = mpimg.imread(sample_img)

plt.imshow(img)

plt.scatter(nucleus_data.iloc[:, 0], nucleus_data.iloc[:, 1], c="red")

plt.scatter(cytoplasm_data.iloc[:, 0], cytoplasm_data.iloc[:, 1], c="blue")

plt.show()
# Imports

import random, cv2, os, sys, shutil

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

import numpy as np

import keras



class image_clustering:



	def __init__(self, folder_path="/kaggle/input/pap-smear-datasets/sipakmed_fci_pap_smear", n_clusters=10, max_examples=None, use_imagenets=False, use_pca=False):

		paths = os.listdir(folder_path)

		if max_examples == None:

			self.max_examples = len(paths)

		else:

			if max_examples > len(paths):

				self.max_examples = len(paths)

			else:

				self.max_examples = max_examples

		self.n_clusters = n_clusters

		self.folder_path = folder_path

		random.shuffle(paths)

		self.image_paths = paths[:self.max_examples]

		self.use_imagenets = use_imagenets

		self.use_pca = use_pca

		print("\n output folders created.")

		os.makedirs("output",exist_ok=True)

		for i in range(self.n_clusters):

			os.makedirs("output\\cluster" + str(i))

		print("\n Object of class \"image_clustering\" has been initialized.")
def load_images(self):

		self.images = []

		for image in self.image_paths:

			self.images.append(cv2.cvtColor(cv2.resize(cv2.imread(self.folder_path + "\\" + image), (224,224)), cv2.COLOR_BGR2RGB))

		self.images = np.float32(self.images).reshape(len(self.images), -1)

		self.images /= 255

		print("\n " + str(self.max_examples) + " images from the \"" + self.folder_path + "\" folder have been loaded in a random order.")
def get_new_imagevectors(self):

		if self.use_imagenets == False:

			self.images_new = self.images

		else:

			if use_imagenets.lower() == "vgg16":

				model1 = keras.applications.vgg19.vgg19(include_top=False, weights="imagenet", input_shape=(224,224,3))

			elif use_imagenets.lower() == "vgg19":

				model1 = keras.applications.vgg19.vgg19(include_top=False, weights="imagenet", input_shape=(224,224,3))

			elif use_imagenets.lower() == "resnet50":

				model1 = keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet", input_shape=(224,224,3))

			elif use_imagenets.lower() == "xception":

				model1 = keras.applications.xception.Xception(include_top=False, weights='imagenet',input_shape=(224,224,3))

			elif use_imagenets.lower() == "inceptionv3":

				keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(224,224,3))

			elif use_imagenets.lower() == "inceptionresnetv2":

				model1 = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(224,224,3))

			elif use_imagenets.lower() == "densenet":

				model1 = keras.applications.densenet.DenseNet201(include_top=False, weights='imagenet', input_shape=(224,224,3))

			elif use_imagenets.lower() == "mobilenetv2":

				model1 = keras.applications.mobilenetv2.MobileNetV2(input_shape=(224,224,3), alpha=1.0, depth_multiplier=1, include_top=False, weights='imagenet', pooling=None)

			else:

				print("\n\n Please use one of the following keras applications only [ \"vgg16\", \"vgg19\", \"resnet50\", \"xception\", \"inceptionv3\", \"inceptionresnetv2\", \"densenet\", \"mobilenetv2\" ] or False")

				sys.exit()



			pred = model1.predict(self.images)

			images_temp = pred.reshape(self.images.shape[0], -1)

			if self.use_pca == False: 

				self.images_new = images_temp

			else: 

				model2 = PCA(n_components=None, random_state=728)

				model2.fit(images_temp)

				self.images_new = model2
def clustering(self):

		model = KMeans(n_clusters=self.n_clusters, n_jobs=-1, random_state=728)

		model.fit(self.images_new)

		predictions = model.predict(self.images_new)

		#print(predictions)

		for i in range(self.max_examples):

			shutil.copy2(self.folder_path+"\\"+self.image_paths[i], "output\cluster"+str(predictions[i]))

		print("\n Clustering complete! \n\n Clusters and the respective images are stored in the \"output\" folder.")
if __name__ == "__main__":



	print("\n\n \t\t START\n\n")



	number_of_clusters = 10 # cluster names will be 0 to number_of_clusters-1



	data_path = "/kaggle/input/pap-smear-datasets/sipakmed_fci_pap_smear" # path of the folder that contains the images to be considered for the clustering (The folder must contain only image files)



	max_examples = 500 # number of examples to use, if "None" all of the images will be taken into consideration for the clustering

	# If the value is greater than the number of images present  in the "data_path" folder, it will use all the images and change the value of this variable to the number of images available in the "data_path" folder. 



	use_imagenets = False

	# choose from: "Xception", "VGG16", "VGG19", "ResNet50", "InceptionV3", "InceptionResNetV2", "DenseNet", "MobileNetV2" and "False" -> Default is: False

	# you have to use the correct spelling! (case of the letters are irrelevant as the lower() function has been used)



	if use_imagenets == False:

		use_pca = False

	else:

		use_pca = False # Make it True if you want to use PCA for dimentionaity reduction -> Default is: False



	temp = image_clustering(data_path, number_of_clusters, max_examples, use_imagenets, use_pca)

	temp.load_img()

	temp.get_new_imagevectors()

	temp.clustering()



	print("\n\n\t\t END\n\n")