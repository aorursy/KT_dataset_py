import numpy as np
import pandas as pd
import math
import cv2
import matplotlib.pyplot as plt
import os
import seaborn as sns
from PIL import Image
import numpy as np
from random import shuffle
data_paths = {}
data_paths['original'] = '/kaggle/input/people-faces/Faces'
data_paths['blurred-0x6'] = '/kaggle/input/people-faces-blurred-0x6/Faces_blurred_0x6'
data_paths['blurred-0x8'] = '/kaggle/input/people-faces-blurred-0x8/Faces_blurred_0x8'
data_paths['pixelated-15'] = '/kaggle/input/people-faces-pixelated-15/Faces_pixelated_15'
data_paths['pixelated-10'] = '/kaggle/input/people-faces-pixelated-10/Faces_pixelated_10'
# Helper function
def plot_images(data_path):
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20,20))
    os.chdir(data_path)
    images = os.listdir()
    shuffle(images)
    plot_count = 0
    for count in range(5):
        ax = axes.flatten()[plot_count]
        ax.imshow(Image.open(images[count]))
        ax.set_xticks([])
        ax.set_yticks([])
        plot_count +=1
    plt.show()
plot_images(data_paths['original'])
plot_images(data_paths['blurred-0x6'])
plot_images(data_paths['blurred-0x8'])
plot_images(data_paths['pixelated-15'])
plot_images(data_paths['pixelated-10'])
record_counts = {}
for name, data_path in data_paths.items():
    os.chdir(data_path)
    record_counts[name] = len(os.listdir())
    
df = pd.DataFrame(record_counts.items(), columns=['dataset', 'image-count'])
print(df.head())
plt.figure(figsize=(8,8))
sns.barplot(df['dataset'], df['image-count'], palette="GnBu_d")
plt.title("Number of Images per Dataset")
plt.show()
os.chdir(data_paths['original'])
images = os.listdir()
male_female = [img.split('_')[1] for img in images]

gender = []
for i in male_female:
    gender.append(int(i))

ax = sns.countplot(gender, palette="BrBG", orient='v')
plt.xlabel('Gender')
labels = ['Male(0)', 'Female(1)']
plt.xticks([0, 1], labels)
plt.title("Gender Distribution")
total = len(gender)
for p in ax.patches:
    percent = '{:.0%}'.format(p.get_height()/total)
    x = p.get_x() + p.get_width()/2 - 0.07
    y = p.get_y() + p.get_height()/2
    ax.annotate(percent, (x,y))
plt.show()