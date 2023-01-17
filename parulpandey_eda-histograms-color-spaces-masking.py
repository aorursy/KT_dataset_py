
import numpy as np 
import pandas as pd 


# For plotting
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import ImageGrid

import cv2
from io import BytesIO
from PIL import Image

# skimage
from skimage.io import imshow, imread, imsave
from skimage.transform import rotate, AffineTransform, warp,rescale, resize, downscale_local_mean
from skimage import color,data
from skimage.exposure import adjust_gamma
from skimage.util import random_noise

#plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)

# 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors


COLORS = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# List files available
print(os.listdir("../input/plant-pathology-2020-fgvc7"))
# Defining data path
IMAGE_PATH = "../input/plant-pathology-2020-fgvc7/images/"

train_df = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")
test_df = pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")


#Training data
print('Training data shape: ', train_df.shape)
train_df.head(5)
# Null values and Data types
print('Train Set')
print(train_df.info())
print('-------------')
print('Test Set')
print(test_df.info())
# Total number of images in the dataset(train+test)
print("Total images in Train set: ",train_df['image_id'].count())
print("Total images in Test set: ",test_df['image_id'].count())

# Categories of Images
classes = ['healthy', 'multiple_diseases', 'rust', 'scab']
print(f"The dataset images belong to the following categories - {classes} ")
for c in classes:
    print(f"The class {c} has {train_df[c].sum()} samples")
healthy = train_df[train_df['healthy'] == 1]['image_id'].to_list()
multiple_diseases = train_df[train_df['multiple_diseases'] == 1]['image_id'].to_list()
rust = train_df[train_df['rust'] == 1]['image_id'].to_list()
scab = train_df[train_df['scab'] == 1]['image_id'].to_list()

diseases = [len(healthy), len(multiple_diseases), len(rust), len(scab)]
diseases

trace = go.Bar(
                    x = classes,
                    y = diseases ,
                    orientation='v',
                    marker = dict(color=COLORS,
                                 line=dict(color='black',width=1)),
                    )
data = [trace]
layout = go.Layout(barmode = "group",title='',width=800, height=500, 
                       xaxis= dict(title='Leaf Categories'),
                       yaxis=dict(title="Count"),
                       showlegend=False)
fig = go.Figure(data = data, layout = layout)
iplot(fig)
train_id = set(train_df.image_id.values )
print(f"Unique Training set Images: {len(train_id)}")
test_id = set(test_df.image_id.values )
print(f"Unique Test set Images: {len(train_id)}")
def duplicacy(df1, df2, image_id):
   
    df1_unique = set(df1['image_id'].values)
    df2_unique = set(df2['image_id'].values)
    images_in_both_dataframes = list(df1_unique.intersection(df2_unique))
    return images_in_both_dataframes
   
    

duplicacy(train_df, test_df, 'image_id')

images = train_df['image_id'].values

# Extract 9 random images from it
random_images = [np.random.choice(images+'.jpg') for i in range(6)]

# Location of the image dir
img_dir = IMAGE_PATH

print('Display Random Images')

# Adjust the size of your images
plt.figure(figsize=(15,10))

# Iterate and plot random images
for i in range(6):
    plt.subplot(2, 3, i + 1)
    img = plt.imread(os.path.join(img_dir, random_images[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
# Adjust subplot parameters to give specified padding
plt.tight_layout()   
def display_images(images, Image_dir, condition):
    random_images = [np.random.choice(images+'.jpg') for i in range(6)]

    print(f"Display {condition} Images")

   # Adjust the size of your images
    plt.figure(figsize=(15,10))

  # Iterate and plot random images
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        img = plt.imread(os.path.join(img_dir, random_images[i]))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    
# Adjust subplot parameters to give specified padding
    plt.tight_layout()   
    
    
healthy_images = np.array(healthy,dtype='object')
images = healthy_images
display_images(images,IMAGE_PATH,'healthy')
rust_images = np.array(rust,dtype='object')
images = rust_images
display_images(images,IMAGE_PATH,'rust')
scab_images = np.array(scab,dtype='object')
images = scab_images
display_images(images,IMAGE_PATH,'scab')
multiple_diseases_images = np.array(multiple_diseases,dtype='object')
images = multiple_diseases_images
display_images(images,IMAGE_PATH,'multiple_diseases')
f = plt.figure(figsize=(16,8))
f.add_subplot(1,2, 1)

sample_img = healthy[0]+'.jpg'
raw_image = plt.imread(os.path.join(img_dir, sample_img))
plt.imshow(raw_image, cmap='gray')
plt.colorbar()
plt.title('Healthy Image')
print(f"Image dimensions:  {raw_image.shape[0],raw_image.shape[1]}")
print(f"Maximum pixel value : {raw_image.max():.1f} ; Minimum pixel value:{raw_image.min():.1f}")
print(f"Mean value of the pixels : {raw_image.mean():.1f} ; Standard deviation : {raw_image.std():.1f}")

f.add_subplot(1,2, 2)

#_ = plt.hist(raw_image.ravel(),bins = 256, color = 'orange',)
_ = plt.hist(raw_image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)
_ = plt.hist(raw_image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)
_ = plt.hist(raw_image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)
_ = plt.xlabel('Intensity Value')
_ = plt.ylabel('Count')
_ = plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'])
plt.show()

f = plt.figure(figsize=(16,8))
f.add_subplot(1,2, 1)

rust_img = rust[0]+'.jpg'
rust_image = plt.imread(os.path.join(img_dir, rust_img))
plt.imshow(rust_image, cmap='gray')
plt.colorbar()
plt.title('Rust Image')
print(f"Image dimensions:  {raw_image.shape[0],raw_image.shape[1]}")
print(f"Maximum pixel value : {raw_image.max():.1f} ; Minimum pixel value:{raw_image.min():.1f}")
print(f"Mean value of the pixels : {raw_image.mean():.1f} ; Standard deviation : {raw_image.std():.1f}")

f.add_subplot(1,2, 2)
#_ = plt.hist(raw_image.ravel(),bins = 256, color = 'orange',)
_ = plt.hist(rust_image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)
_ = plt.hist(rust_image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)
_ = plt.hist(rust_image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)
_ = plt.xlabel('Intensity Value')
_ = plt.ylabel('Count')
_ = plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'])
plt.show()
f = plt.figure(figsize=(16,8))
f.add_subplot(1,2, 1)

scab_img = scab[0]+'.jpg'
scab_image = plt.imread(os.path.join(img_dir, scab_img))
plt.imshow(scab_image, cmap='gray')
plt.colorbar()
plt.title('Scab Image')
print(f"Image dimensions:  {raw_image.shape[0],raw_image.shape[1]}")
print(f"Maximum pixel value : {raw_image.max():.1f} ; Minimum pixel value:{raw_image.min():.1f}")
print(f"Mean value of the pixels : {raw_image.mean():.1f} ; Standard deviation : {raw_image.std():.1f}")

f.add_subplot(1,2, 2)
#source: https://towardsdatascience.com/histograms-in-image-processing-with-skimage-python-be5938962935
#_ = plt.hist(raw_image.ravel(),bins = 256, color = 'orange',)
_ = plt.hist(scab_image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)
_ = plt.hist(scab_image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)
_ = plt.hist(scab_image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)
_ = plt.xlabel('Intensity Value')
_ = plt.ylabel('Count')
_ = plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'])
plt.show()

f = plt.figure(figsize=(16,8))
f.add_subplot(1,2, 1)

multiple_diseases_img = multiple_diseases[0]+'.jpg'
multiple_diseases_image = plt.imread(os.path.join(img_dir, multiple_diseases_img))
plt.imshow(multiple_diseases_image, cmap='gray')
plt.colorbar()
plt.title('Multiple Diseases Image')
print(f"Image dimensions:  {raw_image.shape[0],raw_image.shape[1]}")
print(f"Maximum pixel value : {raw_image.max():.1f} ; Minimum pixel value:{raw_image.min():.1f}")
print(f"Mean value of the pixels : {raw_image.mean():.1f} ; Standard deviation : {raw_image.std():.1f}")

f.add_subplot(1,2, 2)
#source: https://towardsdatascience.com/histograms-in-image-processing-with-skimage-python-be5938962935
#_ = plt.hist(raw_image.ravel(),bins = 256, color = 'orange',)
_ = plt.hist(multiple_diseases_image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)
_ = plt.hist(multiple_diseases_image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)
_ = plt.hist(multiple_diseases_image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)
_ = plt.xlabel('Intensity Value')
_ = plt.ylabel('Count')
_ = plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'])
plt.show()

img_raw = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_1.jpg')

img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB) 
plt.imshow(img)
r, g, b = cv2.split(img)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()



hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_image)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()
boundaries = [([30,0,0],[70,255,255])]
mask = cv2.inRange(hsv_image, (36, 0, 0), (70, 255,255))
result = cv2.bitwise_and(img, img, mask=mask)

plt.figure(figsize=(16,8))
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()

img_raw2 = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_3.jpg')

img2 = cv2.cvtColor(img_raw2, cv2.COLOR_BGR2RGB)
hsv_image2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
plt.imshow(img2)

boundaries = [([30,0,0],[70,255,255])]
mask = cv2.inRange(hsv_image2, (36, 0, 0), (70, 255,255))
result = cv2.bitwise_and(img2, img2, mask=mask)

plt.figure(figsize=(16,8))
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()
