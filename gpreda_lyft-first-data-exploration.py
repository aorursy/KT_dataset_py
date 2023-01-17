!pip install --upgrade pip

!pip install pymap3d==2.1.0

!pip install -U l5kit
import os

import numpy as np

import pandas as pd

from l5kit.data import ChunkedDataset, LocalDataManager

from l5kit.dataset import EgoDataset, AgentDataset

from l5kit.rasterization import build_rasterizer

from l5kit.configs import load_config_data

from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR

from l5kit.geometry import transform_points

from l5kit.data import PERCEPTION_LABELS

from tqdm import tqdm

from collections import Counter

import matplotlib.pyplot as plt

import seaborn as sns 

from matplotlib import animation, rc

from matplotlib.ticker import MultipleLocator

from IPython.display import display, clear_output

import PIL

from IPython.display import HTML



rc('animation', html='jshtml')
os.environ["L5KIT_DATA_FOLDER"] = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"

cfg = load_config_data("/kaggle/input/lyft-config-files/visualisation_config.yaml")

print(cfg)
# local data manager

dm = LocalDataManager()

# set dataset path

dataset_path = dm.require(cfg["val_data_loader"]["key"])

# load the dataset; this is a zarr format, chunked dataset

chunked_dataset = ChunkedDataset(dataset_path)

# open the dataset

chunked_dataset.open()

print(chunked_dataset)
agents = chunked_dataset.agents

agents_df = pd.DataFrame(agents)

agents_df.columns = ["data"]; features = ['centroid', 'extent', 'yaw', 'velocity', 'track_id', 'label_probabilities']



for i, feature in enumerate(features):

    agents_df[feature] = agents_df['data'].apply(lambda x: x[i])

agents_df.drop(columns=["data"],inplace=True)

print(f"agents dataset: {agents_df.shape}")

agents_df.head()
agents_df['cx'] = agents_df['centroid'].apply(lambda x: x[0])

agents_df['cy'] = agents_df['centroid'].apply(lambda x: x[1])
fig, ax = plt.subplots(1,1,figsize=(8,8))

plt.scatter(agents_df['cx'], agents_df['cy'], marker='+')

plt.xlabel('x', fontsize=11); plt.ylabel('y', fontsize=11)

plt.title("Centroids distribution")

plt.show()
agents_df['ex'] = agents_df['extent'].apply(lambda x: x[0])

agents_df['ey'] = agents_df['extent'].apply(lambda x: x[1])

agents_df['ez'] = agents_df['extent'].apply(lambda x: x[2])
sns.set_style('whitegrid')



fig, ax = plt.subplots(1,3,figsize=(16,5))

plt.subplot(1,3,1)

plt.scatter(agents_df['ex'], agents_df['ey'], marker='+')

plt.xlabel('ex', fontsize=11); plt.ylabel('ey', fontsize=11)

plt.title("Extent: ex-ey")

plt.subplot(1,3,2)

plt.scatter(agents_df['ey'], agents_df['ez'], marker='+', color="red")

plt.xlabel('ey', fontsize=11); plt.ylabel('ez', fontsize=11)

plt.title("Extent: ey-ez")

plt.subplot(1,3,3)

plt.scatter(agents_df['ez'], agents_df['ex'], marker='+', color="green")

plt.xlabel('ez', fontsize=11); plt.ylabel('ex', fontsize=11)

plt.title("Extent: ez-ex")

plt.show();
fig, ax = plt.subplots(1,1,figsize=(8,8))

sns.distplot(agents_df['yaw'],color="magenta")

plt.title("Yaw distribution")

plt.show()
agents_df['vx'] = agents_df['velocity'].apply(lambda x: x[0])

agents_df['vy'] = agents_df['velocity'].apply(lambda x: x[1])
fig, ax = plt.subplots(1,1,figsize=(8,8))

plt.title("Velocity distribution")

plt.scatter(agents_df['vx'], agents_df['vy'], marker='.', color="red")

plt.xlabel('vx', fontsize=11); plt.ylabel('vy', fontsize=11)

plt.show();
print("Number of tracks: ", agents_df.track_id.nunique())

print("Entries per track id (first 10): \n", agents_df.track_id.value_counts()[0:10])

                                        
probabilities = agents["label_probabilities"]

labels_indexes = np.argmax(probabilities, axis=1)

counts = []

for idx_label, label in enumerate(PERCEPTION_LABELS):

    counts.append(np.sum(labels_indexes == idx_label))



agents_df = pd.DataFrame()

for count, label in zip(counts, PERCEPTION_LABELS):

    agents_df = agents_df.append(pd.DataFrame({'label':label, 'count':count},index=[0]))

agents_df = agents_df.reset_index().drop(columns=['index'], axis=1)
print(f"agents probabilities dataset: {agents_df.shape}")

agents_df  
f, ax = plt.subplots(1,1, figsize=(10,4))

plt.scatter(agents_df['label'], agents_df['count']+1, marker='*')

plt.xticks(rotation=90, size=8)

plt.xlabel('Perception label')

plt.ylabel(f'Agents count')

plt.title("Agents perception label values count distribution")

plt.grid(True)

ax.set(yscale="log")

plt.show()
scenes = chunked_dataset.scenes

scenes_df = pd.DataFrame(scenes)

scenes_df.columns = ["data"]; features = ['frame_index_interval', 'host', 'start_time', 'end_time']

for i, feature in enumerate(features):

    scenes_df[feature] = scenes_df['data'].apply(lambda x: x[i])

scenes_df.drop(columns=["data"],inplace=True)

print(f"scenes dataset: {scenes_df.shape}")

scenes_df.head()
f, ax = plt.subplots(1,1, figsize=(6,4))

sns.countplot(scenes_df.host)

plt.xlabel('Host')

plt.ylabel(f'Count')

plt.title("Scenes host count distribution")

plt.show()
scenes_df['frame_index_start'] = scenes_df['frame_index_interval'].apply(lambda x: x[0])

scenes_df['frame_index_end'] = scenes_df['frame_index_interval'].apply(lambda x: x[1])

scenes_df.head()
f, ax = plt.subplots(1,1, figsize=(8,8))

spacing = 498

minorLocator = MultipleLocator(spacing)

ax.yaxis.set_minor_locator(minorLocator)

ax.xaxis.set_minor_locator(minorLocator)

plt.xlabel('Start frame index')

plt.ylabel(f'End frame index')

plt.grid(which = 'minor')

plt.title("Frames scenes start and end index (grouped per host)")

sns.scatterplot(scenes_df['frame_index_start'], scenes_df['frame_index_end'], marker='|',  hue=scenes_df['host'])

plt.show()
frames_df = pd.DataFrame(chunked_dataset.frames)

frames_df.columns = ["data"]; features = ['timestamp', 'agent_index_interval', 'traffic_light_faces_index_interval', 

                                          'ego_translation','ego_rotation']

for i, feature in enumerate(features):

    frames_df[feature] = frames_df['data'].apply(lambda x: x[i])

frames_df.drop(columns=["data"],inplace=True)

print(f"frames dataset: {frames_df.shape}")

frames_df.head()
frames_df['dx'] = frames_df['ego_translation'].apply(lambda x: x[0])

frames_df['dy'] = frames_df['ego_translation'].apply(lambda x: x[1])

frames_df['dz'] = frames_df['ego_translation'].apply(lambda x: x[2])
sns.set_style('whitegrid')

plt.figure()



fig, ax = plt.subplots(1,3,figsize=(16,5))



plt.subplot(1,3,1)

plt.scatter(frames_df['dx'], frames_df['dy'], marker='+')

plt.xlabel('dx', fontsize=11); plt.ylabel('dy', fontsize=11)

plt.title("Translations: dx-dy")

plt.subplot(1,3,2)

plt.scatter(frames_df['dy'], frames_df['dz'], marker='+', color="red")

plt.xlabel('dy', fontsize=11); plt.ylabel('dz', fontsize=11)

plt.title("Translations: dy-dz")

plt.subplot(1,3,3)

plt.scatter(frames_df['dz'], frames_df['dx'], marker='+', color="green")

plt.xlabel('dz', fontsize=11); plt.ylabel('dx', fontsize=11)

plt.title("Translations: dz-dx")



fig.suptitle("Ego translations in 2D planes of the 3 components (dx,dy,dz)", size=14)

plt.show();
fig, ax = plt.subplots(1,3,figsize=(16,5))

colors = ['magenta', 'orange', 'darkblue']; labels= ["dx", "dy", "dz"]

for i in range(0,3):

    df = frames_df['ego_translation'].apply(lambda x: x[i])

    plt.subplot(1,3,i + 1)

    sns.distplot(df, hist=False, color = colors[ i ])

    plt.xlabel(labels[i])

fig.suptitle("Ego translations distribution", size=14)

plt.show()
fig, ax = plt.subplots(3,3,figsize=(16,16))

colors = ['red', 'blue', 'green', 'magenta', 'orange', 'darkblue', 'black', 'cyan', 'darkgreen']

for i in range(0,3):

    for j in range(0,3):

        df = frames_df['ego_rotation'].apply(lambda x: x[i][j])

        plt.subplot(3,3,i * 3 + j + 1)

        sns.distplot(df, hist=False, color = colors[ i * 3 + j  ])

        plt.xlabel(f'r[ {i + 1} ][ {j + 1} ]')

fig.suptitle("Ego rotation angles distribution", size=14)

plt.show()
frames_df['tlfii0'] = frames_df['traffic_light_faces_index_interval'].apply(lambda x: x[0])

frames_df['tlfii1'] = frames_df['traffic_light_faces_index_interval'].apply(lambda x: x[1])

sns.set_style('whitegrid')

plt.figure()

fig, ax = plt.subplots(1,1,figsize=(8,8))

plt.scatter(frames_df['tlfii0'], frames_df['tlfii1'], marker='+')

plt.xlabel('Trafic lights faces index interval [0]', fontsize=11); plt.ylabel('Trafic lights faces index interval [1]', fontsize=11)

plt.title("Trafic lights faces index interval")

plt.show()
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

colors = ['cyan', 'darkgreen']

for i in range(0,2):

    df = frames_df['agent_index_interval'].apply(lambda x: x[i])

    plt.subplot(1, 2, i + 1)

    sns.distplot(df, hist=False, color = colors[ i ])

    plt.xlabel(f'agent index interval [ {i} ]')

fig.suptitle("Agent index interval", size=14)

plt.show()
def show_scene_animated(images):



    def animate(i):

        im.set_data(images[i])

 

    fig, ax = plt.subplots()

    im = ax.imshow(images[0])

    

    return animation.FuncAnimation(fig, animate, frames=len(images), interval=60)



def prepare_scene_for_animation(scene_index=20,map_type="py_semantic"):

    cfg["raster_params"]["map_type"] = map_type

    rast = build_rasterizer(cfg, dm)

    dataset = EgoDataset(cfg, chunked_dataset, rast)

    scene_idx = scene_index

    indexes = dataset.get_scene_indices(scene_idx)

    images = []



    for idx in indexes:



        data = dataset[idx]

        im = data["image"].transpose(1, 2, 0)

        im = dataset.rasterizer.to_rgb(im)

        target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

        center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]

        draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)

        clear_output(wait=True)

        images.append(PIL.Image.fromarray(im[::-1]))    

    return images
semantic_images_animation = show_scene_animated(prepare_scene_for_animation(10,"py_semantic"))
HTML(semantic_images_animation.to_jshtml())
semantic_images_animation = show_scene_animated(prepare_scene_for_animation(20,"py_semantic"))
HTML(semantic_images_animation.to_jshtml())
semantic_images_animation = show_scene_animated(prepare_scene_for_animation(30,"py_semantic"))
HTML(semantic_images_animation.to_jshtml())
semantic_images_animation = show_scene_animated(prepare_scene_for_animation(40,"py_semantic"))
HTML(semantic_images_animation.to_jshtml())
satellite_images_animation = show_scene_animated(prepare_scene_for_animation(10, "py_satellite"))
HTML(satellite_images_animation.to_jshtml())
satellite_images_animation = show_scene_animated(prepare_scene_for_animation(20, "py_satellite"))
HTML(satellite_images_animation.to_jshtml())
satellite_images_animation = show_scene_animated(prepare_scene_for_animation(30, "py_satellite"))
HTML(satellite_images_animation.to_jshtml())
satellite_images_animation = show_scene_animated(prepare_scene_for_animation(40, "py_satellite"))
HTML(satellite_images_animation.to_jshtml())