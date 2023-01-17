import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os, json, random



from tensorflow.keras.utils import to_categorical
train_path =  "../input/eurosat-dataset/EuroSAT/train.csv"

val_path = "../input/eurosat-dataset/EuroSAT/validation.csv"

test_path = "../input/eurosat-dataset/EuroSAT/test.csv"
train_df = pd.read_csv(train_path)

train_df.drop(columns=train_df.columns[0],inplace=True)

train_df.head()
val_df = pd.read_csv(val_path)

val_df.drop(columns=val_df.columns[0],inplace=True)

val_df.head()
test_df = pd.read_csv(test_path)

test_df.drop(columns=test_df.columns[0],inplace=True)

test_df.head()
with open("../input/eurosat-dataset/EuroSAT/label_map.json","r") as file:

    class_names_encoded = json.load(file)

    

class_names = list(class_names_encoded.keys())

class_names
_, train_labels_count = np.unique(train_df['Label'],return_counts=True)



train_count_df = pd.DataFrame(data = train_labels_count)

train_count_df['ClassName'] = class_names

train_count_df.columns = ['Count','ClassName']

train_count_df.set_index('ClassName',inplace=True)

train_count_df.head()
train_count_df.plot.bar()

plt.title("Distribution of images per class")

plt.ylabel("Count");
plt.pie(train_count_df.Count,

       explode = (0,0,0,0,0,0,0,0,0,0),

       labels = class_names,

       autopct="%1.2f%%")

plt.axis('equal');
_, val_labels_count = np.unique(val_df['Label'], return_counts=True)



val_count_df = pd.DataFrame(data = val_labels_count)

val_count_df['ClassName'] = class_names

val_count_df.columns = ['Count','ClassName']

val_count_df.set_index('ClassName',inplace=True)

val_count_df.head()
val_count_df.plot.bar()

plt.title("Distribution of images per class")

plt.ylabel("Count");
plt.pie(val_count_df.Count,

       explode = (0,0,0,0,0,0,0,0,0,0),

       labels = class_names,

       autopct="%1.2f%%")

plt.axis('equal');
_, test_labels_count = np.unique(test_df['Label'], return_counts=True)



test_count_df = pd.DataFrame(data = test_labels_count)

test_count_df['ClassName'] = class_names

test_count_df.columns = ['Count','ClassName']

test_count_df.set_index('ClassName',inplace=True)

test_count_df.head()
test_count_df.plot.bar()

plt.title("Distribution of images per class")

plt.ylabel("Count");
plt.pie(test_count_df.Count,

       explode = (0,0,0,0,0,0,0,0,0,0),

       labels = class_names,

       autopct="%1.2f%%")

plt.axis('equal');
train_labels = to_categorical(train_df['Label'],num_classes=len(class_names))

train_labels.shape
classTotals = train_labels.sum(axis=0)

classWeight = {}



for i in range(len(classTotals)):

    classWeight[i] = classTotals.max()/classTotals[i]

    pass



classWeight
classWeight = json.dumps(str(classWeight))



with open("classWeight.json","w") as f:

    f.write(classWeight)