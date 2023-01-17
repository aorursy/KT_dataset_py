import numpy as np 

import xml.etree.ElementTree as ET # for parsing XML



import matplotlib.pyplot as plt # to show images

from PIL import Image # to read images



import os
#Boilerplate code to generate equally sized images, cropped from the source to only include the regions containing dogs

def grabImageAndAnnotation(strBreed):

    imgFolder = '../input/images/Images/'

    imgDir = [(imgFolder + s) for s in os.listdir(imgFolder) if strBreed in s][0]

    imgPics = os.listdir(imgDir)

    

    annotFolder = '../input/annotations/Annotation/'

    annotDir = [(annotFolder + s) for s in os.listdir(annotFolder) if strBreed in s][0]

    annotFiles = os.listdir(annotDir)

    

    boxDict = extractBoundingBoxes(annotDir,annotFiles)

    imgDict = {s:Image.open(os.path.join(imgDir,s)) for s in imgPics}

    return (boxDict, imgDict)

    

def extractBoundingBoxes(dirName,filesNames):

    boundingBoxes = {}

    for path in filesNames:

        tree = ET.parse(os.path.join(dirName,path))

        root = tree.getroot()

        #Traverse down nodes to get to bounding box data

        obj = root.find('object')

        bbox = obj.find('bndbox')

        xmin = int(bbox.find('xmin').text)

        xmax = int(bbox.find('xmax').text)

        ymin = int(bbox.find('ymin').text)

        ymax = int(bbox.find('ymax').text)

        boundingBoxes[path] = [xmin,xmax, ymin,ymax]

    return boundingBoxes



#resize to preset width and height whilst maintaining the same aspect ratio

#centre image and add black letterboxes if necessary

def generateCentredResizedCroppedImages(imageDict, boxDict, w, h):

    resizedList = []

    for images in boxDict.keys():

        bbox = boxDict[images]

        srcWidth = bbox[1] - bbox[0] 

        srcHeight = bbox[3] - bbox[2] 

        scaleFactor = min(w / srcWidth, h / srcHeight)

        newWidth = int(scaleFactor * srcWidth)

        newHeight = int(scaleFactor * srcHeight)

        newX = (w - newWidth) // 2

        newY = (h - newHeight) // 2

        

        blankImg = Image.new('RGB', (w,h))

        srcCropped = imageDict[images + '.jpg'].crop((bbox[0], bbox[2], bbox[1], bbox[3]))

        srcCropped.load()

        srcResized = srcCropped.resize((newWidth, newHeight), Image.LANCZOS)

        blankImg.paste(srcResized, (newX, newY))

        resizedList.append(blankImg)

    return resizedList
(boxDict, imgDict) = grabImageAndAnnotation('beagle')
max_width = max([arr[1] - arr[0] for arr in boxDict.values()])

max_height = max([arr[3] - arr[2] for arr in boxDict.values()])

print('max_width:{}, max_height:{}'.format(max_width,max_height))
target_width = 300

target_height = 300

resized = generateCentredResizedCroppedImages(imgDict, boxDict, target_width, target_height)
plt.imshow(resized[110])
resized_mat = [np.array(img) for img in resized]

avg_image = np.average(resized_mat, axis = 0)

plt.imshow(avg_image.astype(np.uint8))
#Prep data for PCA by subtracting overall mean pixel values for each pixel in every image in order to obtain a vector with a mean of 0

#Also flatten the 2d image to a 1d vector

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

resized_mat_flattened  = scaler.fit_transform(np.reshape(resized_mat,[-1, target_width * target_height * 3]))
from sklearn.model_selection import train_test_split

X_train, X_test= train_test_split(resized_mat_flattened,test_size=0.2, random_state=42)
from sklearn.decomposition import PCA

eigen_beagle = PCA( copy = True)

eigen_beagle.fit(X_train)
eigen_vectors = eigen_beagle.components_

eigen_vectors_img = eigen_vectors.reshape([-1,target_height,target_width,3])



plt.figure(figsize = (18,9))

nplots = 15

for i in range (1,nplots + 1):

    plt.subplot(nplots / 5,5,i)

    #The eigenvectors are not clamped to any particular value

    #Scaled to a value of [0,1] using the maximum and minimum of each eigenvector

    v_min = np.min(eigen_vectors_img[i])

    v_max = np.max(eigen_vectors_img[i])

    scaled_eigen_vector = (eigen_vectors_img[i] - v_min ) / (v_max - v_min)

    plt.imshow(scaled_eigen_vector)

    plt.title('eigen vector:{}'.format(i))

plt.tight_layout()
nComponents = 192 #Number of eigenvectors used for reconstruction

test_img = X_test[3]

fitted = eigen_beagle.transform([test_img])[0] #Apply PCA transform

fitted[nComponents:len(fitted)] = 0 #remove higher components 

reconstructed =  scaler.inverse_transform(eigen_beagle.inverse_transform(fitted))



plt.figure(figsize = (10,5))

plt.subplot(1,2,1)

plt.title('Reconstructed')

plt.imshow(reconstructed.reshape(target_width,target_height,3).astype(int))

plt.subplot(1,2,2)

plt.imshow(scaler.inverse_transform(test_img).reshape(target_width,target_height,3).astype(int))

plt.title('Original')
def compute_database_coeff(pca_model, data):

    return pca_model.transform(data)



def min_euclidean_difference(coeff_list, weights):

    dist = np.sum(np.power(np.subtract(coeff, weights), 2), axis = 1)

    return (np.min(dist), np.argmin(dist))
coeff = compute_database_coeff(eigen_beagle, X_train)



(diff, train_index) = min_euclidean_difference(coeff, fitted)



print('Minimum difference from training set:{}\nCorresponding training image index:{}'.format(diff, train_index))



plt.figure(figsize = (10,8))

plt.subplot(2,2,1)

plt.imshow(reconstructed.reshape(target_width,target_height,3).astype(int))

plt.title('Reconstructed')

plt.subplot(2,2,2)

plt.imshow(scaler.inverse_transform(X_train[train_index]).reshape(target_width,target_height,3).astype(int))

plt.title('Closest match in training set')



plt.subplot(2,2,3)

plt.imshow(scaler.inverse_transform(test_img).reshape(target_width,target_height,3).astype(int))

plt.title('Actual image')



plt.tight_layout()
from numpy.linalg import eig



class PCA2D:

    def __init__(self):

        self.e_vectors = []

        self.e_values = []

    

    #X (n,width,height,channels)

    def fit(self, samples):

        # Based on "Two-Dimensional PCA: A New Approach to Appearance-Based Face Representation and Recognition"

        # By Yang et al (2004).

        avg_image = np.average(samples, axis = 0)

        diff = samples - avg_image

        diff_row_major = np.transpose(diff, (0,3,1,2))

        diffTransposed = np.transpose(diff, (0,3,2,1))

        #this matrix multiply accounts for majority of the computation time

        mult = np.matmul(diffTransposed, diff_row_major)

        mult_sum = np.sum(mult, axis = 0)

        g_t = 1/len(samples) * mult_sum

        (values, vectors) = eig(g_t) #already lists eigenvalues in descending value

        

        self.e_vectors = vectors

        self.e_values = values

    

    def transform(self, image):

        #separate RGB values into separate 2d arrays

        img_row_major = image.transpose(2,0,1)

        #Get component values

        y_components = np.matmul(img_row_major, self.e_vectors)

        return y_components

    

    def inverse_transform(self, y_components, n_components = None):

        n_comp = len(y_components[0][0]) if n_components is None else n_components

        

        y_components_truncated = y_components[:,:, :n_comp]

        eigen_vectors_truncated = self.e_vectors[:,:,:n_comp]

        

        reconstructed_row_major = np.matmul(y_components_truncated, 

                                            np.transpose(eigen_vectors_truncated,(0,2,1)))

        

        reconstructed_img = reconstructed_row_major.transpose(1,2,0)

        return reconstructed_img
samples = resized_mat[0:100]



pca2d = PCA2D()

pca2d.fit(samples)
y = pca2d.transform(resized_mat[110])

r_img = pca2d.inverse_transform(y, n_components = 35)

plt.imshow(r_img.astype(int))
def cumSum(vector):

    return np.cumsum((vector / np.sum(vector)))
rcs = cumSum(pca2d.e_values[0])[0:40]

gcs = cumSum(pca2d.e_values[1])[0:40]

bcs = cumSum(pca2d.e_values[2])[0:40]
plt.figure(figsize = (10,5))

plt.plot(rcs, 'r')

plt.plot(gcs, 'g')

plt.plot(bcs, 'b')

plt.plot([0,40],[0.95,0.95], 'k--')

plt.title('Cumulative sum of eigenvalues across R,G & B channels')
test_images =  resized_mat[100:]

training_y_weights = [pca2d.transform(img) for img in samples]
def generate_diff_list(raw_y_weights, n_weights):

    trunc_y = np.array(raw_y_weights)[:,:,:,0:n_weights]

    y_max_diff_list = []

    y_min_diff_list = []

    

    for current_y in trunc_y:

        y_diff = np.subtract(current_y, trunc_y)

        y_diff_sqr = np.power(y_diff, 2)

        y_diff_sum = np.sum(y_diff_sqr, axis = (1,2,3))

        #Get maximum difference between this image and the other training samples

        y_max_diff_list.append(np.sqrt(max(y_diff_sum)))

        #Get minimum difference between this image and the other training samples

        y_diff_sum_exclusive = y_diff_sum[y_diff_sum != 0]

        y_min_diff_list.append(np.sqrt(min(y_diff_sum_exclusive)))

    return (y_min_diff_list, y_max_diff_list)
(y_min_diff_list, y_max_diff_list) = generate_diff_list(training_y_weights , 35)
plt.figure(figsize = (10,5))

plt.subplot(1,2,1)

plt.hist(y_min_diff_list)

plt.title('Min differences (RSME)\n[Between training images from other training images]')

plt.subplot(1,2,2)

plt.hist(y_max_diff_list)

plt.title('Max differences (RSME)\n[Between training images from other training images]')

plt.tight_layout()
test_y_weights = [pca2d.transform(img) for img in test_images]
def generate_min_diff(test_weight, raw_training_y_weights, n_weights):

    trunc_y_test = np.array(test_weight)[:,:,:,0:n_weights]

    trunc_y_train = np.array(raw_training_y_weights)[:,:,:,0:n_weights]

    y_diff_list = []

    for current_y in trunc_y_test:

        y_diff = np.subtract(current_y, trunc_y_train)

        y_diff_sqr = np.power(y_diff, 2)

        y_diff_sum = np.sum(y_diff_sqr, axis = (1,2,3))

        y_diff_list.append(np.sqrt(min(y_diff_sum)))

    return y_diff_list
test_y_diff_list = generate_min_diff(test_y_weights, training_y_weights, 35)
plt.hist(test_y_diff_list)

plt.title('Minimum difference (RSME)\n[Between new test images from training set]')
(basset_boxDict, basset_imgDict) = grabImageAndAnnotation('basset')

basset_resized = generateCentredResizedCroppedImages(basset_imgDict, basset_boxDict, target_width, target_height)

basset_mat = [np.array(img) for img in basset_resized]

plt.imshow(basset_mat[10])
basset_y_weights = [pca2d.transform(img) for img in basset_mat]

basset_y_diff = generate_min_diff(basset_y_weights, training_y_weights, 35)
plt.hist(basset_y_diff)

plt.title('Minimum difference (RSME)\n[Between Basset Hound test images from Beagle training set]')