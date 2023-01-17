# Load dependencies
import urllib.parse
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

PATH = '../input/landmark-retrieval-2020/'

# Get labels
labels = pd.read_csv(PATH+'train.csv', index_col='id')['landmark_id'].to_dict()

# Download csn and get classes
url = 'https://s3.amazonaws.com/google-landmark/metadata/train_label_to_category.csv'
CLASSES = pd.read_csv(url, index_col = 'landmark_id', encoding='latin', engine='python')['category'].to_dict()
# Input image
image_id = 'cd41bf948edc0340' # try changing this id to see a different landmark and its name
image_path = f'{PATH}/train/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg'

# Read file and decode
image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image, channels=3)
image = tf.cast(image, tf.int64)

# Identify the landmark id and name
landmark_id = labels[image_id]
landmark_name = urllib.parse.unquote(CLASSES[labels[image_id]].replace('http://commons.wikimedia.org/wiki/Category:', ''))
title = f'{landmark_id}: {landmark_name}'
print (title)

# Show the image with title
plt.figure(figsize=(20,10))
plt.axis('off')
plt.imshow(image)
plt.title(title, fontsize=16)
plt.show()