# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#Load the initial data
train_data = pd.read_csv("../input/train/train.csv")


# Any results you write to the current directory are saved as output.
vertex_xs = []
vertex_ys = []
bounding_confidences = []
bounding_importance_fracs = []
dominant_blues = []
dominant_greens = []
dominant_reds = []
dominant_pixel_fracs = []
dominant_scores = []
label_descriptions = []
label_scores = []
face_annotations = []
label_annotations = []
text_annotations = []
nf_count = 0
nl_count = 0
for pet in train_data.PetID:
    try:
        with open('../input/train_metadata/' + pet + '-1.json', 'r') as f:
            data = json.load(f)
        face_annotations.append(data.get('faceAnnotations', []))
        text_annotations.append(data.get('textAnnotations', []))
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        vertex_xs.append(vertex_x)
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
        vertex_ys.append(vertex_y)
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
        bounding_confidences.append(bounding_confidence)
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        bounding_importance_fracs.append(bounding_importance_frac)
        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        dominant_blues.append(dominant_blue)
        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        dominant_greens.append(dominant_green)
        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        dominant_reds.append(dominant_red)
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        dominant_pixel_fracs.append(dominant_pixel_frac)
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        dominant_scores.append(dominant_score)
        if data.get('labelAnnotations'):
            label_annotations.append(data['labelAnnotations'])
            label_description = data['labelAnnotations'][0]['description']
            label_descriptions.append(label_description)
            label_score = data['labelAnnotations'][0]['score']
            label_scores.append(label_score)
        else:
            nl_count += 1
            label_annotations.append([])
            label_descriptions.append('nothing')
            label_scores.append(-1)
    except FileNotFoundError:
        nf_count += 1
        vertex_xs.append(-1)
        vertex_ys.append(-1)
        bounding_confidences.append(-1)
        bounding_importance_fracs.append(-1)
        dominant_blues.append(-1)
        dominant_greens.append(-1)
        dominant_reds.append(-1)
        dominant_pixel_fracs.append(-1)
        dominant_scores.append(-1)
        label_annotations.append([])
        label_descriptions.append('nothing')
        label_scores.append(-1)
        face_annotations.append([])
        text_annotations.append([])

print(nf_count)
print(nl_count)
train_data.loc[:, 'vertex_x'] = vertex_xs
train_data.loc[:, 'vertex_y'] = vertex_ys
train_data.loc[:, 'bounding_confidence'] = bounding_confidences
train_data.loc[:, 'bounding_importance'] = bounding_importance_fracs
train_data.loc[:, 'dominant_blue'] = dominant_blues
train_data.loc[:, 'dominant_green'] = dominant_greens
train_data.loc[:, 'dominant_red'] = dominant_reds
train_data.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
train_data.loc[:, 'dominant_score'] = dominant_scores
train_data.loc[:, 'label_description'] = label_descriptions
train_data.loc[:, 'label_score'] = label_scores
train_data.sample(5)
count = 0
pet_ids = []
index = -1
for pet_id in train_data.PetID:
    index += 1
    try:
        im = Image.open('../input/train_images/%s-1.jpg' % pet_id)
        width, height = im.size
        vertex_y = vertex_ys[index]
        vertex_x = vertex_xs[index]
        if vertex_y < height - 10 or vertex_x < width - 10:
            pet_ids.append(pet_id)
            count += 1
    except:
        pass
    

print(f"{count} pets have their profile picture's crop hint significantly different than the whole image itself")
from collections import Counter
train_data.loc[:, 'num_faces'] = list(map(lambda x: len(x), face_annotations))
sns.countplot(x='num_faces', data=train_data)
Counter(train_data['num_faces'])
sns.catplot(x='num_faces', y='AdoptionSpeed', data=train_data, kind='bar')
plt.title("AdoptionSpeed based on number of faces in image")
plt.show()
def dog_or_cat(label_annotation):
    if len(label_annotation) > 0:
        desc = label_annotation[0]['description']
        if desc == 'cat' or desc == 'dog':
            return True
    return False

animal_scores = []
animal_topics = []
indices = []
for label_annotation in label_annotations:
    score = -1
    topic = -1
    index = 0
    for label in label_annotation:
        if label['description'] == 'dog' or label['description'] == 'cat':
            score = label['score']
            topic = label['topicality']
            indices.append(index)
            break
        index += 1
    if score == -1:
        indices.append(-1)
    animal_scores.append(score)
    animal_topics.append(topic)

train_data.loc[:, 'dominant_animal_label'] = list(map(lambda x: dog_or_cat(x), label_annotations))
train_data.loc[:, 'animal_scores'] = animal_scores
train_data.loc[:, 'animal_topic'] = animal_topics
sns.catplot(x='dominant_animal_label', y='AdoptionSpeed', data=train_data, kind='bar')
plt.title("AdoptionSpeed based on whether first(strongest) label is Dog/Cat")
plt.show()
print("Count of Indices the Animal Label Occurs In: ", Counter(indices))
print("Count of Dominant Animal Label (True/False): ", Counter(train_data['dominant_animal_label']))
train_data.loc[:, 'animal_index'] = indices
sns.catplot(x='animal_index', y='AdoptionSpeed', data=train_data, kind='bar')
plt.title("AdoptionSpeed based on Index of Animal Label (-1 == Not Found)")
plt.show()
sns.catplot(x="AdoptionSpeed", y="animal_scores", data=train_data, kind="strip")
plt.show()
sns.countplot(x="AdoptionSpeed", data=train_data).set_title("Distribution of AdoptionSpeed (All Data)")
plt.show()
sns.countplot(x="AdoptionSpeed", data=train_data.loc[train_data['animal_scores'] == -1]).set_title("Distribution of AdoptionSpeed with no Animal Label")
plt.show()
sns.catplot(x="AdoptionSpeed", y="animal_scores", data=train_data.loc[train_data['animal_scores'] != -1], kind="strip")
plt.title("Animal Label Score distribution for various AdoptionSpeeds")
plt.show()
sns.catplot(x="AdoptionSpeed", y="label_score", data=train_data.loc[train_data['label_score'] != -1], kind="strip")
plt.title("First Label Score distribution for various AdoptionSpeeds")
plt.show()
#Check if has textAnnnotations

has_text_annotations = list(map(lambda x: len(x), text_annotations))
print("True=Has Text Annotations in JSON: ", Counter(has_text_annotations))
train_data.loc[:, 'has_text'] = has_text_annotations
sns.catplot(x="has_text", y="AdoptionSpeed", data=train_data, kind="bar")
plt.title("AdoptionSpeed based on whether text is present in the profile")
plt.show()

def mapToDescLen(text_annotation):
    if (len(text_annotation) == 0):
        return 0
    return len(text_annotation[0]['description'])

text_length = list(map(lambda x: mapToDescLen(x), text_annotations))
train_data.loc[:, 'text_length'] = text_length

sns.catplot(x='AdoptionSpeed', y='text_length', data=train_data, kind='strip')
plt.title("Plot of text length in JSON vs AdoptionSpeed")
plt.show()

sns.catplot(x='AdoptionSpeed', y='text_length', data=train_data, kind='bar')
plt.title("Mean Lengths per AdoptionSpeed of Text in Profile Picture")
plt.show()