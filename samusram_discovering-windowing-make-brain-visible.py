from IPython.display import Image

Image('../input/illustrations/preproc.jpg')
!pip install hdbscan
import os

import numpy as np

import pydicom

import pandas as pd

from random import sample

from tqdm import tqdm_notebook as tqdm

import hdbscan

from scipy.spatial.distance import jensenshannon

from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score

import pickle

import random

random.seed(1)

from numpy.random import seed

seed(1)

import matplotlib.pyplot as plt

%matplotlib inline
train_folder = '../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/'
train_df = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')

train_df.head()
img_labels_df = train_df[train_df['Label'] == 1].copy()

img_labels_df['Label_String'] = img_labels_df['ID'].map(lambda x: x.split('_')[-1])

img_labels_df['Image'] = img_labels_df['ID'].map(lambda x: 'ID_' + x.split('_')[1])

img_2_labels = img_labels_df.groupby('Image')['Label_String'].agg(list)
plt.hist(img_2_labels.map(len))

plt.xlabel('Number of conditions per positive image')

plt.ylabel('Count')

plt.title('Counts of labels per positive image')
print(f"Label any is present in {100*sum(img_2_labels.map(lambda x: 'any' in x))/len(img_2_labels):.2f}% of positive images.")
img_2_labels = img_2_labels.map(lambda x: [i for i in x if i != 'any'])

img_2_labels = img_2_labels.map(lambda x: ', '.join(x))

img_2_labels.head()
train_img_names = os.listdir('../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/')

print(f'There are {len(train_img_names)} train images.')
train_img_names_subsample = sample(train_img_names, 1000)
def get_img(img_name, folder=train_folder):

    pydicom_filedataset = pydicom.read_file(os.path.join(folder, img_name))

    return pydicom_filedataset.pixel_array



def vizualize_tuple(imgs, img_names, grid_size=3):

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size+4, grid_size+4))

    fig.tight_layout()

    for img_i, img in enumerate(imgs):

        ax = axes[img_i//grid_size, img_i%grid_size]

        ax.imshow(img, cmap='bone')

        ax.axis('off')

        ax.set_title(f"{img_names[img_i]}:\n{img_2_labels.get(img_names[img_i].replace('.dcm', ''), '')}")
img_names_subsample = sample(train_img_names_subsample, 9)

vizualize_tuple([get_img(img_name) for img_name in img_names_subsample], img_names_subsample)
imgs_to_compare = ['ID_df5ae8f49.dcm', 'ID_8cb74b318.dcm']



def plot_img_pair(imgs, suptitle='', img_names=imgs_to_compare):

    fig, axes = plt.subplots(1, 2, figsize=(7, 6))

    fig.tight_layout()

    for img_i, img in enumerate(imgs):

        ax = axes[img_i]

        ax.imshow(img, cmap='bone')

        ax.axis('off')

        ax.set_title(f"{img_names[img_i]}:\n{img_2_labels.get(img_names[img_i].replace('.dcm', ''), '')}")

    plt.suptitle(suptitle, fontsize=15)

        

plot_img_pair([get_img(img_name) for img_name in imgs_to_compare])
def plot_intensity_hists(imgs, bins=None, img_names=imgs_to_compare):

    min_intesity = min([np.min(img) for img in imgs])

    max_intesity = max([np.max(img) for img in imgs])

    for img_i, img in enumerate(imgs):

        plt.figure(figsize=(4, 4))

        plt.hist(img.flatten(), bins=bins)

        plt.xlabel('Image intensities', fontsize=11)

        plt.ylabel('Pixels count', fontsize=11)

        plt.xlim((min_intesity, max_intesity))

        plt.title(f"{img_names[img_i]}:\n{img_2_labels.get(img_names[img_i].replace('.dcm', ''), '')}", fontsize=14)

        

plot_intensity_hists([get_img(img_name) for img_name in imgs_to_compare])
plot_img_pair([get_img(img_name) for img_name in imgs_to_compare], 'Initial Images')



def clip_negatives(img, threshold=0):

    img[img < threshold] = threshold

    return img



plot_img_pair([clip_negatives(get_img(img_name)) for img_name in imgs_to_compare], 'Without Negative Intensities')
plot_intensity_hists([clip_negatives(get_img(img_name)) for img_name in imgs_to_compare], bins=50)
plot_img_pair([get_img(img_name) for img_name in imgs_to_compare], 'Initial Images')

def clip_positives(img, threshold=1500):

    img[img > threshold] = threshold

    return img



lb = 700

ub = 1500

clipped_imgs = [clip_positives(clip_negatives(get_img(img_name), lb), ub) for img_name in imgs_to_compare]

plot_img_pair(clipped_imgs, f'Intensities between {lb} and {ub}')
plot_intensity_hists(clipped_imgs, bins=50)
plot_img_pair([get_img(img_name) for img_name in imgs_to_compare], 'Initial Images')



lb = 900

ub = 1200

clipped_imgs = [clip_positives(clip_negatives(get_img(img_name), lb), ub) for img_name in imgs_to_compare]

plot_img_pair(clipped_imgs, f'Intensities between {lb} and {ub}')
img_names_diverse = ['ID_c96b7ba2a.dcm', 'ID_8cb74b318.dcm']

imgs_diverse = [get_img(img_name) for img_name in img_names_diverse]

plot_img_pair(imgs_diverse, img_names=img_names_diverse)

plot_intensity_hists(imgs_diverse, img_names=img_names_diverse)
def generate_hist_vec(img_name, bins=list(range(-50, 1500, 25))):

    img = get_img(img_name)

    return np.histogram(img[(img > -50) & (img < 1500)].flatten(), bins=bins)[0]



histogram_vectors = []

for img_name in tqdm(train_img_names_subsample, desc='Generation histogram vectors..'):

    histogram_vectors.append(generate_hist_vec(img_name))
histograms_df = pd.DataFrame(np.array(histogram_vectors), columns=[f'bin_{i}' for i in range(len(histogram_vectors[0]))])
# clusterer = DBSCAN(eps=0.1, min_samples=3, metric=jensenshannon)

clusterer = hdbscan.HDBSCAN(metric=jensenshannon)

clusterer.fit(histograms_df)
print(f'There are {max(clusterer.labels_) + 1} clusters.')
print(f"""Sizes of the clusters:

{pd.Series(clusterer.labels_).value_counts()}.""")
def get_cluster_sample(cluster_i, sample_size=9):

    cluster_img_names = [img for img, is_in in zip(train_img_names_subsample, clusterer.labels_==cluster_i) 

                               if is_in]

    cluster_sample = sample(cluster_img_names, min(sample_size, len(cluster_img_names)))

    return cluster_sample



                

def check_cluster_vizually(cluster_sample, lb=-4000, ub=5000, hist=True): 

    clipped_imgs = [clip_positives(clip_negatives(get_img(img_name), lb), ub) for img_name in cluster_sample]

                    

    def vizualize_nine_hists(imgs):

        min_intesity = min([np.min(img) for img in imgs])

        max_intesity = max([np.max(img) for img in imgs])

        fig, axes = plt.subplots(3, 3, figsize=(7, 7))

        for img_i, img in enumerate(imgs):

            ax = axes[img_i//3, img_i%3]

            ax.hist(img.flatten(), bins=20)

            ax.set_xlim((min_intesity, max_intesity))

            ax.set_yticks([], [])

        fig.suptitle('Histograms of pixel intensities')

        

    vizualize_tuple(clipped_imgs, cluster_sample)

    if hist:

        vizualize_nine_hists(clipped_imgs)

    

    

cluster_samples = [get_cluster_sample(cluster_i) for cluster_i in range(max(clusterer.labels_) + 1)]

for cluster_i, cluster_sample in enumerate(cluster_samples):

    plt.figure(figsize=(9, 1))

    plt.plot(np.arange(20), np.ones(20))

    plt.title(f'Cluster {cluster_i}')

    check_cluster_vizually(cluster_sample, hist=False)
cluster_2_intensity_limits = []
check_cluster_vizually(cluster_samples[0], lb=930, ub=1150)
check_cluster_vizually(cluster_samples[0], lb=830, ub=930, hist=False)

check_cluster_vizually(cluster_samples[0], lb=1150, ub=1250, hist=False)
cluster_2_intensity_limits.append((930, 1150))
check_cluster_vizually(cluster_samples[1], lb=920, ub=1200)
check_cluster_vizually(cluster_samples[1], lb=820, ub=920, hist=False)

check_cluster_vizually(cluster_samples[1], lb=1200, ub=1300, hist=False)
cluster_2_intensity_limits.append((920, 1200))
check_cluster_vizually(cluster_samples[2], lb=5, ub=300)
check_cluster_vizually(cluster_samples[2], lb=-95, ub=5, hist=False)

check_cluster_vizually(cluster_samples[2], lb=300, ub=400, hist=False)
cluster_2_intensity_limits.append((5, 300))
check_cluster_vizually(cluster_samples[3], lb=1500, ub=2000)
check_cluster_vizually(cluster_samples[3], lb=970, ub=1300)
check_cluster_vizually(cluster_samples[3], lb=870, ub=970, hist=False)

check_cluster_vizually(cluster_samples[3], lb=1300, ub=1400, hist=False)
cluster_2_intensity_limits.append((970, 1300))
check_cluster_vizually(cluster_samples[4], lb=1400, ub=2100)
check_cluster_vizually(cluster_samples[4], lb=900, ub=1300)
check_cluster_vizually(cluster_samples[4], lb=800, ub=900, hist=False)

check_cluster_vizually(cluster_samples[4], lb=1300, ub=1400, hist=False)
cluster_2_intensity_limits.append((900, 1300))
non_noise_bool_index = clusterer.labels_ != -1

histograms_df = histograms_df[non_noise_bool_index]

labels = clusterer.labels_[non_noise_bool_index]
scaler = StandardScaler()

histograms_df_scaled = scaler.fit_transform(histograms_df)
train_df, test_df, train_labels, test_labels = train_test_split(histograms_df_scaled, labels, test_size=0.2, stratify=labels)
svm = LinearSVC(multi_class='crammer_singer')

svm.fit(train_df, train_labels)
test_predictions = svm.predict(test_df)
print(f"Average weighted F1 score on test data is: {f1_score(test_labels, test_predictions, average='weighted')}.")
with open('scaler.pkl', 'wb') as f:

    pickle.dump(scaler, f)

    

with open('svm_cluster_type.pkl', 'wb') as f:

    pickle.dump(svm, f)
used_set = set(train_img_names_subsample)

unseen_img_names = [img for img in train_img_names if img not in used_set]

unseen_img_sample = sample(unseen_img_names, 9)



histogram_vectors = [generate_hist_vec(img) for img in unseen_img_sample]
used_set = set(train_img_names_subsample)

unseen_img_names = [img for img in train_img_names if img not in used_set]

unseen_img_sample = sample(unseen_img_names, 16)



histogram_vectors = scaler.transform([generate_hist_vec(img) for img in unseen_img_sample])

img_cluster_classes = svm.predict(histogram_vectors)

clipping_limits = [cluster_2_intensity_limits[class_i] for class_i in img_cluster_classes]

clipped_imgs = [clip_positives(clip_negatives(get_img(img_name), lb), ub) for img_name, (lb, ub) in zip(unseen_img_sample,

                                                                                                       clipping_limits)]

vizualize_tuple(clipped_imgs, unseen_img_sample, 4)
def plot_pie_per_cluster_type(cluster_class_i):

    all_known_cluster_imgs = get_cluster_sample(cluster_class_i, sample_size=float('inf'))

    all_deceaseas = []

    for img in all_known_cluster_imgs:

        all_deceaseas.extend(img_2_labels.get(img.replace('.dcm', ''), 'nothing').split(', '))

    all_deceaseas_counts = pd.Series(all_deceaseas).value_counts()

    

    fig, ax = plt.subplots(figsize=(7, 7))

    # credits: https://stackoverflow.com/questions/6170246/how-do-i-use-matplotlib-autopct

    def make_autopct(values):

        def my_autopct(pct):

            total = sum(values)

            val = int(round(pct*total/100.0))

            return '{p:.0f}%  ({v:d})'.format(p=pct,v=val)

        return my_autopct

    ax.pie(all_deceaseas_counts, labels=all_deceaseas_counts.index, autopct=make_autopct(all_deceaseas_counts), shadow=True, startangle=90)

    ax.axis('equal')

    plt.title(f'Distribution of deceases per class {cluster_class_i}', fontsize=15)
for cluster_class_i in range(max(clusterer.labels_)+1):

    plot_pie_per_cluster_type(cluster_class_i)