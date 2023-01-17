%%capture

import os

for dirname, _, filename in os.walk("../input"):

  for files in filename:

    print(os.path.join(dirname, files))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import tensorflow as tf

%matplotlib inline

import cv2

from tqdm import tqdm_notebook as tqdm
train_data = pd.DataFrame(pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv"))

test_data = pd.DataFrame(pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv"))
print("Training data shape : = {}".format(train_data.shape))

print("Test data shape : = {}".format(test_data.shape))
test_data.head()
image_folder_path = "../input/plant-pathology-2020-fgvc7/images/"
arr = train_data["image_id"]

train_images = [i for i in arr]  



arr = test_data["image_id"]

test_images = [i for i in arr]
def load_image(image_id) : 

  image_path = image_folder_path +image_id +".jpg"

  image = cv2.imread(image_path) 

  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  return image



def resize(image):

  image = cv2.resize(image, (800, 800))

  return image
def extract_classes(s):

  """

  s can be either of the four classes mentioned above.

  """ 

  t = train_data[train_data[s] == 1] 

  arr = t["image_id"]

  images = [i for i in tqdm(arr)]

  train_images = [load_image(i) for i in tqdm(images)]

  return train_images



classes = ["healthy", "multiple_diseases", "rust", "scab"] 
count_healthy = len(train_data[train_data["healthy"] == 1])

count_diseased = len(train_data[train_data["multiple_diseases"] == 1])

count_rust = len(train_data[train_data["rust"] == 1])

count_scab = len(train_data[train_data["scab"] == 1])



print(count_healthy)

print(count_diseased)

print(count_rust)

print(count_scab)

print(count_healthy + count_diseased + count_rust +  count_scab)
# observe number of cases present in each class

labels = ["Healthy", "Multiple Diseased", "Rust", "Scab"]

counts = [count_healthy, count_diseased, count_rust, count_scab]

explode = (0.05, 0.05, 0.05, 0.05)

fig, ax = plt.subplots(figsize = (20, 12))

ax.pie(counts, explode = explode, labels = labels, shadow = True, startangle = 90)

ax.axis("equal") # equal aspect ratio ensures pie graph is drawn as circle
red , green, blue = [], [], []
healthy = extract_classes("healthy")

for image in healthy :

    mean_red = np.mean(image[:,:,0])

    mean_green = np.mean(image[:,:,1])

    mean_blue = np.mean(image[:,:,2])

    

    red.append(mean_red)

    green.append(mean_green)

    blue.append(mean_blue)

    

healthy_image_1 = healthy[100]

healthy_image_2 = healthy[200]

healthy_image_3 = healthy[300]

del healthy # free memory



md = extract_classes("multiple_diseases")

for image in md : 

    mean_red = np.mean(image[:,:,0])

    mean_green = np.mean(image[:,:,1])

    mean_blue = np.mean(image[:,:,2])

    

    red.append(mean_red)

    green.append(mean_green)

    blue.append(mean_blue)

md_image_1 = md[1]

md_image_2 = md[5]

md_image_3 = md[10]

del md # free memory



rust = extract_classes("rust")

for image in rust : 

    mean_red = np.mean(image[:,:,0])

    mean_green = np.mean(image[:,:,1])

    mean_blue = np.mean(image[:,:,2])

    

    red.append(mean_red)

    green.append(mean_green)

    blue.append(mean_blue)

rust_image_1 = rust[10]

rust_image_2 = rust[20] 

rust_image_3 = rust[30]

del rust # free memory



scab = extract_classes("healthy")

for image in scab : 

    mean_red = np.mean(image[:,:,0])

    mean_green = np.mean(image[:,:,1])

    mean_blue = np.mean(image[:,:,2])

    

    red.append(mean_red)

    green.append(mean_green)

    blue.append(mean_blue)

scab_image_1 = scab[10]

scab_image_2 = scab[20]

scab_image_3 = scab[30] 

del scab # free memory



image_collection = [healthy_image_1, healthy_image_2, healthy_image_3, 

                   md_image_1, md_image_2, md_image_3,

                   rust_image_1, rust_image_2, rust_image_3,

                   scab_image_1, scab_image_2, scab_image_3]   
fig, ax = plt.subplots(nrows = 4, ncols = 3, figsize = (25, 15))

for i in range(12):

    ax[i//3, i%3].imshow(image_collection[i]) 
# red channel plot

range_of_spread = max(red) - min(red)

plt.figure(figsize = (12, 8))

plt.rc('font', weight='bold')

sns.set_style("whitegrid")

fig = sns.distplot(red,  hist = True, kde = True, label = "Red Channel intensities", color = "r")

fig.set(xlabel = "Mean red channel intensities observed in each image (Sample size = 1000)", ylabel = "Probability Density")

plt.legend()

print("The range of spread = {:.2f}".format(range_of_spread))
# Green channel plot

range_of_spread = max(green) - min(green)

plt.figure(figsize = (12, 8))

plt.rc('font', weight='bold')

sns.set_style("whitegrid")

fig = sns.distplot(green,  hist = True, kde = True, label = "Green Channel intensities", color = "g")

fig.set(xlabel = "Mean green channel intensities observed in each image (Sample size = 1000)", ylabel = "Probability Density")

plt.legend()

print("The range of spread = {:.2f}".format(range_of_spread))
# Blue channel plot

range_of_spread = max(blue) - min(blue)

plt.figure(figsize = (12, 8))

plt.rc('font', weight='bold')

sns.set_style("whitegrid")

fig = sns.distplot(blue,  hist = True, kde = True, rug = False, label = "Blue Channel intensities", color = "b")

fig.set(xlabel = "Mean blue channel intensities observed in each image (Sample size = 1000)", ylabel = "Probability Density")

plt.legend()

print("The range of spread = {:.2f}".format(range_of_spread))
sample_image = rust_image_1

plt.figure(figsize = (12, 8))

plt.imshow(sample_image)
def non_local_means_denoising(image) : 

    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    return denoised_image
denoised_image = non_local_means_denoising(sample_image)



plt.figure(figsize = (12, 8))

plt.subplot(1,2,1)

plt.imshow(sample_image, cmap = "gray")

plt.grid(False)

plt.title("Normal Image")



plt.subplot(1,2,2)  

plt.imshow(denoised_image, cmap = "gray")

plt.grid(False)

plt.title("Denoised image")    

# Automatically adjust subplot parameters to give specified padding.

plt.tight_layout()
def sobel_edge_detection(image):

  """

  Using Sobel filter



  Sobel filter takes the following arguments : 

  1. Original Image

  2. Depth of the destination image

  3. Order of derivative x

  4. Order of derivative y

  5. Kernel size for convolutions



  f(Image, depth, order_dx, order_dy, kernel_size) 

  """

  sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = 5)

  sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = 5)

  return sobel_x, sobel_y
s_img_x, s_img_y = sobel_edge_detection(denoised_image)



plt.figure(figsize = (12, 8))

plt.subplot(2,2,1)

plt.imshow(sample_image, cmap = "gray")

plt.grid(False)

plt.title("Sample Image")



plt.subplot(2,2,2)

plt.imshow(denoised_image, cmap = "gray")

plt.grid(False)

plt.title("Denoised Image")



plt.subplot(2,2,3)

plt.imshow(s_img_x, cmap = "gray")

plt.grid(False)

plt.title("Sobel X filtered Image")



plt.subplot(2,2,4)

plt.imshow(s_img_y, cmap = "gray")

plt.grid(False)

plt.title("Sobel Y filtered Image")



# Automatically adjust subplot parameters to give specified padding.

plt.tight_layout()
from collections import deque

def canny_edge_detection(image):

  edges = cv2.Canny(image, 170, 200) 

  return edges



def primary_roi(original_image, edge_image):

  edge_coordinates = deque()

  for i in tqdm(range(edge_image.shape[0])):

    for j in range(edge_image.shape[1]):

      if edge_image[i][j] != 0 :

        edge_coordinates.append((i, j))

  

  min_row = edge_coordinates[np.argsort([coordinate[0] for coordinate in edge_coordinates])[0]][0]

  max_row = edge_coordinates[np.argsort([coordinate[0] for coordinate in edge_coordinates])[-1]][0]

  min_col = edge_coordinates[np.argsort([coordinate[1] for coordinate in edge_coordinates])[0]][1]

  max_col = edge_coordinates[np.argsort([coordinate[1] for coordinate in edge_coordinates])[-1]][1]

  

  new_image = original_image.copy()

  new_edge_image = edge_image.copy()

  

  new_image[min_row - 10 : min_row + 10, min_col : max_col] = [255, 0, 0]

  new_image[max_row - 10 : max_row + 10, min_col : max_col] = [255, 0, 0]

  new_image[min_row : max_row , min_col - 10 : min_col + 10] = [255, 0, 0]

  new_image[min_row : max_row , max_col - 10 : max_col + 10] = [255, 0, 0]



  new_edge_image[min_row - 10 : min_row + 10, min_col : max_col] = [255]

  new_edge_image[max_row - 10 : max_row + 10, min_col : max_col] = [255]

  new_edge_image[min_row : max_row , min_col - 10 : min_col + 10] = [255]

  new_edge_image[min_row : max_row , max_col - 10 : max_col + 10] = [255]



  roi_image = new_image[min_row : max_row, min_col : max_col]

  edge_roi_image = new_edge_image[min_row : max_row, min_col : max_col]

  

  

  return roi_image, edge_roi_image
plt.figure(figsize = (12, 8))

plt.subplot(1,2,1)

plt.imshow(sample_image, cmap = "gray")

plt.grid(False)

plt.title("Denoised Image")



edge_image = canny_edge_detection(sample_image) 



plt.subplot(1,2,2)

plt.imshow(edge_image, cmap = "gray")

plt.grid(False)

plt.title("Canny Edge Image")

# Automatically adjust subplot parameters to give specified padding.

plt.tight_layout()
roi_image, edge_roi_image = primary_roi(sample_image, edge_image)



plt.figure(figsize = (12, 8))

plt.subplot(1,2,1)

plt.imshow(roi_image, cmap = "gray")

plt.grid(False)

plt.title("ROI Image")



plt.subplot(1,2,2)

plt.imshow(edge_roi_image, cmap = "gray")

plt.grid(False)

plt.title("Edge ROI Image")  

# Automatically adjust subplot parameters to give specified padding.

plt.tight_layout()
def histogram_equalization(roi_image):

  image_ycrcb = cv2.cvtColor(roi_image, cv2.COLOR_RGB2YCR_CB)

  y_channel = image_ycrcb[:, :, 0] # apply histogram equalization on this channel

  cr_channel = image_ycrcb[:, :, 1]

  cb_channel = image_ycrcb[:, :, 2]

  # local histogram equalization

  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

  equalized = clahe.apply(y_channel)

  equalized_image = cv2.merge([equalized, cr_channel, cb_channel])

  equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_YCR_CB2RGB)

  return equalized_image
equalized_roi_image = histogram_equalization(roi_image)



plt.figure(figsize = (12, 8))

plt.subplot(1,2,1)

plt.imshow(roi_image, cmap = "gray")

plt.grid(False)

plt.title("ROI Image")



plt.subplot(1,2,2)

plt.imshow(equalized_roi_image, cmap = "gray")

plt.grid(False)

plt.title("Histogram Equalized ROI Image")  

# Automatically adjust subplot parameters to give specified padding.

plt.tight_layout()
otsu_threshold, otsu_image = cv2.threshold(cv2.cvtColor(equalized_roi_image, cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)



plt.figure(figsize = (12, 8))

plt.subplot(1,2,1)

plt.imshow(equalized_roi_image, cmap = "gray")

plt.grid(False)

plt.title("equalized_roi_image")



plt.subplot(1,2,2)

plt.imshow(otsu_image, cmap = "gray")

plt.grid(False)

plt.title("Otsu's Thresholded Image")  

# Automatically adjust subplot parameters to give specified padding.



plt.tight_layout()
def segmentation(image, k, attempts) : 

    vectorized = np.float32(image.reshape((-1, 3)))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    res , label , center = cv2.kmeans(vectorized, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    center = np.uint8(center)

    res = center[label.flatten()]

    segmented_image = res.reshape((image.shape))

    return segmented_image
plt.figure(figsize = (12, 8))

plt.subplot(2,2,1)

plt.imshow(equalized_roi_image, cmap = "gray")

plt.grid(False)

plt.title("Histogram Equalized Image")



segmented_image = segmentation(equalized_roi_image, 3, 10) # k = 3, attempt = 10

plt.subplot(2,2,2)

plt.imshow(segmented_image, cmap = "gray")

plt.grid(False)

plt.title("Segmented Image with k = 3")



segmented_image = segmentation(equalized_roi_image, 4, 10) # k = 4, attempt = 10

plt.subplot(2,2,3)

plt.imshow(segmented_image, cmap = "gray")

plt.grid(False)

plt.title("Segmented Image with k = 4")



segmented_image = segmentation(equalized_roi_image, 5, 10) # k = 5, attempt = 10

plt.subplot(2,2,4)

plt.imshow(segmented_image, cmap = "gray")

plt.grid(False)

plt.title("Segmented Image with k = 5")