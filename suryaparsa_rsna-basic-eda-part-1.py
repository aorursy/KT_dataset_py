import numpy as np

import pandas as pd

import pydicom

import cv2

import os

import matplotlib.pyplot as plt

import seaborn as sns
input_folder = '../input/rsna-intracranial-hemorrhage-detection/'
path_train_img = input_folder + 'stage_1_train_images/'

path_test_img = input_folder + 'stage_1_test_images/'
train_df = pd.read_csv(input_folder + 'stage_1_train.csv')

train_df.head()
# extract subtype

train_df['sub_type'] = train_df['ID'].apply(lambda x: x.split('_')[-1])

# extract filename

train_df['file_name'] = train_df['ID'].apply(lambda x: '_'.join(x.split('_')[:2]) + '.dcm')

train_df.head()
train_df.shape
print("Number of train images availabe:", len(os.listdir(path_train_img)))
train_df[train_df['sub_type'] == 'epidural']['Label'].value_counts()
sns.countplot(x='Label', data=train_df[train_df['sub_type'] == 'epidural'])

plt.show()
train_df[train_df['sub_type'] == 'intraparenchymal']['Label'].value_counts()
sns.countplot(x='Label', data=train_df[train_df['sub_type'] == 'intraparenchymal'])

plt.show()
train_df[train_df['sub_type'] == 'intraventricular']['Label'].value_counts()
sns.countplot(x='Label', data=train_df[train_df['sub_type'] == 'intraventricular'])

plt.show()
train_df[train_df['sub_type'] == 'subarachnoid']['Label'].value_counts()
sns.countplot(x='Label', data=train_df[train_df['sub_type'] == 'subarachnoid'])

plt.show()
train_df[train_df['sub_type'] == 'subdural']['Label'].value_counts()
sns.countplot(x='Label', data=train_df[train_df['sub_type'] == 'subdural'])

plt.show()
train_df[train_df['sub_type'] == 'any']['Label'].value_counts()
sns.countplot(x='Label', data=train_df[train_df['sub_type'] == 'any'])

plt.show()
train_final_df = pd.pivot_table(train_df.drop(columns='ID'), index="file_name", \

                                columns="sub_type", values="Label")

train_final_df.head()
plt.figure(figsize=(16, 6))



graph = sns.countplot(x="sub_type", hue="Label", data=(train_df))

graph.set_xticklabels(graph.get_xticklabels(),rotation=90)

plt.show()