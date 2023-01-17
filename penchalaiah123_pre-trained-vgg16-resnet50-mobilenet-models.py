import os
import random
import seaborn as sns
import cv2

# General packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL
import IPython.display as ipd
import glob
import h5py
import plotly.graph_objs as go
import plotly.express as px
from PIL import Image
from tempfile import mktemp

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, LinearAxis, Range1d
from bokeh.models.tools import HoverTool
from bokeh.palettes import BuGn4
from bokeh.plotting import figure, output_notebook, show
from bokeh.transform import cumsum
from math import pi

output_notebook()

from IPython.display import Image, display
import warnings
warnings.filterwarnings("ignore")
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
print(os.listdir('../input/landmark-recognition-2020/'))
BASE_PATH = '../input/landmark-recognition-2020'

TRAIN_DIR = f'{BASE_PATH}/train'
TEST_DIR = f'{BASE_PATH}/test'

print('Reading data...')
train = pd.read_csv(f'{BASE_PATH}/train.csv')
sub = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')
print('Reading data completed')
train.head()
sub.head()
train.shape[0]
sub.shape
landmarks = len(train['landmark_id'].unique())
landmarks
print('Top few landmark_ids by count')

z = train.landmark_id.value_counts().head(10).to_frame()
z.reset_index(inplace=True)
z.columns=['landmark_id','count']
z.landmark_id = z.landmark_id.apply(lambda x: f'id_{x}')

z.style.background_gradient(cmap='Oranges')
# displaying only top 30 landmark
landmark = train.landmark_id.value_counts()
landmark_df = pd.DataFrame({'landmark_id':landmark.index, 'frequency':landmark.values}).head(30)

landmark_df['landmark_id'] =   landmark_df.landmark_id.apply(lambda x: f'landmark_id_{x}')

fig = px.bar(landmark_df, x="frequency", y="landmark_id",color='landmark_id', orientation='h',
             hover_data=["landmark_id", "frequency"],
             height=1000,
             title='Number of images per landmark_id (Top 30 landmark_ids)')
fig.show()
import PIL
from PIL import Image, ImageDraw


def display_images(images, title=None): 
    f, ax = plt.subplots(5,5, figsize=(18,22))
    if title:
        f.suptitle(title, fontsize = 30)

    for i, image_id in enumerate(images):
        image_path = os.path.join(TRAIN_DIR, f'{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg')
        image = Image.open(image_path)
        
        ax[i//5, i%5].imshow(image) 
        image.close()       
        ax[i//5, i%5].axis('off')

        landmark_id = train[train.id==image_id.split('.')[0]].landmark_id.values[0]
        ax[i//5, i%5].set_title(f"ID: {image_id.split('.')[0]}\nLandmark_id: {landmark_id}", fontsize="12")

    plt.show() 
samples = train.sample(25).id.values
display_images(samples)
samples = train[train.landmark_id == 138982].sample(25).id.values
display_images(samples)
from collections import Counter
landmark_counts = dict(Counter(train['landmark_id']))
landmark_dict = {'landmark_id': list(landmark_counts.keys()), 'count': list(landmark_counts.values())}

landmark_count_df = pd.DataFrame.from_dict(landmark_dict)
landmark_count_sorted = landmark_count_df.sort_values('count', ascending = False)
landmark_count_sorted.head(20)
fig_count = px.histogram(landmark_count_df, x = 'landmark_id', y = 'count')
fig_count.update_layout(
    title_text='Distribution of Landmarks',
    xaxis_title_text='Landmark ID',
    yaxis_title_text='Count'
)

fig_count.show()
from tensorflow.keras.applications import(
                vgg16,
                resnet50,
                mobilenet,
                inception_v3)
vgg_model = vgg16.VGG16(weights = 'imagenet')
resnet_model = resnet50.ResNet50(weights = 'imagenet')
mobilenet_model = mobilenet.MobileNet(weights = 'imagenet') 
train_list = glob.glob('../input/landmark-recognition-2020/train/*/*/*/*')
test_list = glob.glob('../input/landmark-recognition-2020/test/*/*/*/*')
train_list
test_list
filename = '../input/landmark-recognition-2020/train/1/1/1/11172998c813fe6f.jpg'
original = image.load_img(filename,target_size=(224,224))
print('PIL image size',original.size)
plt.imshow(original)
plt.show()
from tensorflow.keras.preprocessing.image import img_to_array
numpy_image = img_to_array(original)
plt.imshow(np.uint8(numpy_image))
plt.show()
print('numpy array size',numpy_image.shape)
image_batch = np.expand_dims(numpy_image, axis=0)
print('image batch size', image_batch.shape)
plt.imshow(np.uint8(image_batch[0]))
# prepare the image for the VGG model
from keras.applications.imagenet_utils import decode_predictions
processed_image = vgg16.preprocess_input(image_batch.copy())
# get the predicted probabilities for each class
predictions = vgg_model.predict(processed_image)
# print predictions
# convert the probabilities to class labels
# we will get top 5 predictions which is the default
label_vgg = decode_predictions(predictions)
# print VGG16 predictions
for prediction_id in range(len(label_vgg[0])):
    print(label_vgg[0][prediction_id])

from keras.applications.resnet50 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
img = image.load_img(filename,target_size=(224,224))
img = image.img_to_array(img)
img = np.expand_dims(img,axis=0)
img = preprocess_input(img)
preds = resnet_model.predict(img)
print( decode_predictions(preds, top=1)[0])
pred = mobilenet_model.predict(img)
print(decode_predictions(pred))
