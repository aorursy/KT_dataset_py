%%capture

# Install facenet-pytorch

!pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-1.0.1-py3-none-any.whl



# Copy model checkpoints to torch cache so they are loaded automatically by the package

!mkdir -p /tmp/.cache/torch/checkpoints/

!cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-logits.pth /tmp/.cache/torch/checkpoints/vggface2_DG3kwML46X.pt

!cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-features.pth /tmp/.cache/torch/checkpoints/vggface2_G5aNV2VSMn.pt
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path

from facenet_pytorch import MTCNN

import torch

import torch.nn as nn

import torch.nn.functional as F

import cv2

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid

from PIL import Image

from tqdm.notebook import tqdm

from time import time

import shutil



import warnings

warnings.filterwarnings("ignore")



# https://www.kaggle.com/hmendonca/kaggle-pytorch-utility-script

from kaggle_pytorch_utility_script import *



seed_everything(42)
# See github.com/timesler/facenet-pytorch:

from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face



device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Running on device: {device}')
margin = 80

image_size = 150



# Load face detector

mtcnn = MTCNN(keep_all=False, select_largest=False, post_process=False,

              device=device, min_face_size=100,

              margin=margin, image_size=image_size).eval()



# Load facial recognition model

resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
test_videos = '/kaggle/input/deepfake-detection-challenge/test_videos'

train_sample_videos = '/kaggle/input/deepfake-detection-challenge/train_sample_videos'

train_videos = '/kaggle/input/deepfake'

faces_path = '/kaggle/working/faces'



!mkdir -p {faces_path}
import glob

face_files = []

meta = pd.DataFrame()

for jsn in glob.glob(f'{train_videos}/metadata*.json'):

    df = pd.read_json(jsn).transpose()

    chunk = int(jsn[len(f'{train_videos}/metadata'):-len('.json')])

    df['chunk'] = chunk

    meta = pd.concat([meta, df])

    path = f'{train_videos}/DeepFake{chunk:02}/DeepFake{chunk:02}'

    print(meta.shape, jsn, path)

    assert os.path.isdir(path)

    # symlink all images to 'faces_path'

    !ls {path} | xargs -IN ln -sf {path}/N {faces_path}/

    faces = [f'{faces_path}/{vid[:-4]}.jpg' for vid in df[df.label == 'REAL'].index.tolist()]

    face_files.extend(faces)

print(f'Found {len(face_files)} real videos in {len(meta.chunk.unique())} folders')

assert len(face_files) == len(meta[meta.label == 'REAL'])
# get missing images from their fakes

missing_files, recovered_files = [], []

df = []

for idx in tqdm(meta[meta.label == 'REAL'].index,

                total=sum(meta.label == 'REAL')):

    real_image = f'{faces_path}/{idx[:-4]}.jpg'

    if not os.path.isfile(real_image):

#         print(idx, real_image)

        for fidx in meta.loc[meta.original == idx].index:

            fake_image = f'{faces_path}/{fidx[:-4]}.jpg'

            if os.path.isfile(fake_image):

#                 print(idx, fake_image)

                # reuse the first valid fake face as the face for the real video

                !ln -sf {fake_image} {real_image}

                assert os.path.isfile(real_image)

                recovered_files.append(idx)

                break

        if not os.path.isfile(real_image):

            missing_files.append(idx)

print('Recovered', len(recovered_files), 'files, but still missing', len(missing_files),

      'in the total of', len(face_files))



face_files = [f for f in face_files if os.path.isfile(f)]

print('New total:', len(face_files))
# def save_frame(file, folder):

#     reader = cv2.VideoCapture(file)

#     _, image = reader.read()

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     pilimg = Image.fromarray(image)

#     with torch.no_grad():

# #         boxes, probs = mtcnn.detect(pilimg)

#         face_image = mtcnn(pilimg)

#         try:

#             if face_image is not None:

#                 pilface = Image.fromarray(face_image.byte().numpy().transpose([1,2,0]))

#                 imgfile = f'{Path(file).stem}.jpg'

#                 pilface.save(Path(folder)/imgfile)

#         except Exception as e:

#             print(e)

#             return
# folder = '/kaggle/working/faces'

# Path(folder).mkdir(parents=True, exist_ok=True)

# for file in tqdm(list_files):

#     save_frame(file, folder)



# face_files = [str(x) for x in Path(folder).glob('*')]
from torchvision.transforms import ToTensor



tf_img = lambda i: ToTensor()(i).unsqueeze(0)

embeddings = lambda input: resnet(input)
list_embs = []

with torch.no_grad():

    for face in tqdm(face_files):

        t = tf_img(Image.open(face)).to(device)

        e = embeddings(t).squeeze().cpu().tolist()

        list_embs.append(e)
df = pd.DataFrame({'face': face_files[:len(list_embs)], 'embedding': list_embs})

df['video'] = df.face.apply(lambda x: f'{Path(x).stem}.mp4')

df['chunk'] = df.video.apply(lambda x: int(meta.loc[x].chunk))

df = df[['video', 'face', 'chunk', 'embedding']]

df
from sklearn.decomposition import PCA



def scatter_thumbnails(data, images, zoom=0.12, colors=None):

    assert len(data) == len(images)



    # reduce embedding dimentions to 2

    x = PCA(n_components=2).fit_transform(data) if len(data[0]) > 2 else data



    # create a scatter plot.

    f = plt.figure(figsize=(22, 15))

    ax = plt.subplot(aspect='equal')

    sc = ax.scatter(x[:,0], x[:,1], s=4)

    _ = ax.axis('off')

    _ = ax.axis('tight')



    # add thumbnails :)

    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    for i in range(len(images)):

        image = plt.imread(images[i])

        im = OffsetImage(image, zoom=zoom)

        bboxprops = dict(edgecolor=colors[i]) if colors is not None else None

        ab = AnnotationBbox(im, x[i], xycoords='data',

                            frameon=(bboxprops is not None),

                            pad=0.02,

                            bboxprops=bboxprops)

        ax.add_artist(ab)

    return ax



_ = scatter_thumbnails(df.embedding.tolist(), df.face.tolist())

plt.title('Facial Embeddings - Principal Component Analysis')

plt.show()
%%time

from sklearn.manifold import TSNE

# PCA first to speed it up

x = PCA(n_components=50).fit_transform(df['embedding'].tolist())

x = TSNE(perplexity=50,

         n_components=3).fit_transform(x)



_ = scatter_thumbnails(x, df.face.tolist(), zoom=0.06)

plt.title('3D t-Distributed Stochastic Neighbor Embedding')

plt.show()
# !pip install -q hdbscan

# import hdbscan

import sklearn.cluster as cluster
def plot_clusters(data, algorithm, *args, **kwds):

    labels = algorithm(*args, **kwds).fit_predict(data)

    palette = sns.color_palette('deep', np.max(labels) + 1)

    colors = [palette[x] if x >= 0 else (0,0,0) for x in labels]

    ax = scatter_thumbnails(x, df.face.tolist(), 0.06, colors)

    plt.title(f'Clusters found by {algorithm.__name__}')

    return labels



# clusters = plot_clusters(x, hdbscan.HDBSCAN, alpha=1.0, min_cluster_size=2, min_samples=1)

clusters = plot_clusters(x, cluster.DBSCAN, n_jobs=-1, eps=1.0, min_samples=1)

df['cluster'] = clusters
# clusters and the number of images on each one of them

ids, counts = np.unique(clusters, return_counts=True)

_ = pd.DataFrame(counts, index=ids).hist(bins=len(ids), log=True)
from annoy import AnnoyIndex



f = len(df['embedding'][0])

t = AnnoyIndex(f, metric='euclidean')

ntree = 50



for i, vector in enumerate(df['embedding']):

    t.add_item(i, vector)

_  = t.build(ntree)
def get_similar_images_annoy(img_index, n=8, max_dist=1.0):

    vid, face  = df.iloc[img_index, [0, 1]]

    similar_img_ids, dist = t.get_nns_by_item(img_index, n+1, include_distances=True)

    similar_img_ids = [s for s,d in zip(similar_img_ids, dist) if d <= max_dist][1:]  # first item is always its own video

    return vid, face, df.iloc[similar_img_ids], dist
def get_sample_n_similar(sample_idx):

    vid, face, similar, distances = get_similar_images_annoy(sample_idx)

    

    fig = plt.figure(figsize=(15, 7))

    gs = fig.add_gridspec(2, 6)

    ax1 = fig.add_subplot(gs[0:2, 0:2])

    ax2 = fig.add_subplot(gs[0, 2])

    ax3 = fig.add_subplot(gs[0, 3])

    ax4 = fig.add_subplot(gs[0, 4])

    ax5 = fig.add_subplot(gs[0, 5])

    ax6 = fig.add_subplot(gs[1, 2])

    ax7 = fig.add_subplot(gs[1, 3])

    ax8 = fig.add_subplot(gs[1, 4])

    ax9 = fig.add_subplot(gs[1, 5])

    axx = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]

    for ax in axx:

        ax.set_axis_off()

    list_plot = [face] + similar['face'].values.tolist()

    list_cluster = [df.iloc[sample_idx]['cluster']] + similar['cluster'].values.tolist()

    for ax, face, cluster, dist in zip(axx, list_plot, list_cluster, distances):

        ax.imshow(plt.imread(face))

        ax.set_title(f'{face.split("/")[-1][:-4]} @{dist:.2f}\ncluster:{cluster}') # show video filename and distance
# display samples and their nearest neighbors

for i in np.random.choice(len(df), 8, replace=False):

    get_sample_n_similar(i)
chunks = df.groupby('cluster').chunk.nunique().to_frame('n_chunks')

chunks = chunks.merge(df.groupby('cluster').video.nunique().to_frame('n_videos'),

                     left_index=True, right_index=True).sort_values(by='n_chunks')



print(f'{sum(chunks.n_chunks > 1)} clusters are spread across more than one data chunk')

chunks
# sample 2 images of each ['cluster', 'chunk'] pair 

mixed_clusters = chunks[(chunks.n_chunks > 1) & (chunks.n_videos < 100)].index.values

video_samples = df[df.cluster.isin(mixed_clusters)].groupby(['cluster', 'chunk']).face

video_samples = video_samples.agg(['min', 'max']).reset_index()



for cluster in mixed_clusters:

    chunk_samples = video_samples[video_samples.cluster == cluster]

    fig, axes = plt.subplots(2, len(chunk_samples), figsize=(len(chunk_samples)*2, 4))

    print(f'Cluster {cluster} with {len(chunk_samples)} chunks')

    for i, (idx, row) in enumerate(chunk_samples.iterrows()):

        axes[0, i].imshow(plt.imread(row['min']))

        axes[0, i].set_axis_off()

        axes[0, i].set_title(f"""Data chunk: {row.chunk}

{row['min'].split('/')[-1][:-4]}.mp4""")

        if row['max'] != row['min']:

            axes[1, i].imshow(plt.imread(row['max']))

            axes[1, i].set_title(f"""{row['max'].split('/')[-1][:-4]}.mp4""")

        axes[1, i].set_axis_off()

    plt.show()
chunks = df.groupby('cluster').chunk.unique().to_frame('chunks')

chunks = chunks[chunks.chunks.apply(lambda c: len(c)) > 1] # filter non-unique clusters

chunks_df = pd.DataFrame(range(50), columns=['chunk'])

chunks_df['n_nonunique_clusters'] = 0

chunks_df['nonunique_clusters'] = [[] for _ in range(50)]

for i in range(50):

    chunks_df.loc[i, 'n_nonunique_clusters'] = len(chunks[chunks.chunks.apply(lambda chunks: i in chunks)])

    chunks_df.loc[i, 'nonunique_clusters'].extend(chunks[chunks.chunks.apply(lambda chunks: i in chunks)].index.tolist())

chunks_df.sort_values(by='n_nonunique_clusters').head()
# clean up working dir

if not is_interactive():

    shutil.rmtree(Path(faces_path))
# save face clusters

df.to_csv('face_clusters.csv.zip', index=False)

df[['video', 'chunk', 'cluster']].to_feather('face_clusters.feather')

df
meta['cluster'] = -1

meta.loc[df.video, 'cluster'] = df.cluster.values



# propagate real video clusters to their fake versions

for index, real in tqdm(meta[~meta.cluster.isnull()].iterrows(),

                        total=len(meta[~meta.cluster.isnull()])):

    meta.loc[meta.original == index, 'cluster'] = int(real.cluster)



meta.reset_index().to_feather('metadata.feather')

meta[~meta.cluster.isnull()]
# exported files

!ls -sh face_clusters*

!ls -sh metadata*