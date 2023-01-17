%matplotlib inline

import os

from glob import glob

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from skimage.io import imread

DATA_DIR = '../input/20180706181348/'

!ls {DATA_DIR}
# point cloud stuff

from collections import defaultdict

import sys

sys_byteorder = ('>', '<')[sys.byteorder == 'little']

ply_dtypes = dict([

    (b'int8', 'i1'),

    (b'char', 'i1'),

    (b'uint8', 'u1'),

    (b'uchar', 'b1'),

    (b'uchar', 'u1'),

    (b'int16', 'i2'),

    (b'short', 'i2'),

    (b'uint16', 'u2'),

    (b'ushort', 'u2'),

    (b'int32', 'i4'),

    (b'int', 'i4'),

    (b'uint32', 'u4'),

    (b'uint', 'u4'),

    (b'float32', 'f4'),

    (b'float', 'f4'),

    (b'float64', 'f8'),

    (b'double', 'f8')

])

valid_formats = {'ascii': '', 'binary_big_endian': '>',

                 'binary_little_endian': '<'}



def read_ply(filename):

    with open(filename, 'rb') as ply:



        if b'ply' not in ply.readline():

            raise ValueError('The file does not start whith the word ply')

        # get binary_little/big or ascii

        fmt = ply.readline().split()[1].decode()

        # get extension for building the numpy dtypes

        ext = valid_formats[fmt]



        line = []

        dtypes = defaultdict(list)

        count = 2

        points_size = None

        mesh_size = None

        while b'end_header' not in line and line != b'':

            line = ply.readline()



            if b'element' in line:

                line = line.split()

                name = line[1].decode()

                size = int(line[2])

                if name == "vertex":

                    points_size = size

                elif name == "face":

                    mesh_size = size



            elif b'property' in line:

                line = line.split()

                # element mesh

                if b'list' in line:

                    mesh_names = ['n_points', 'v1', 'v2', 'v3']



                    if fmt == "ascii":

                        # the first number has different dtype than the list

                        dtypes[name].append(

                            (mesh_names[0], ply_dtypes[line[2]]))

                        # rest of the numbers have the same dtype

                        dt = ply_dtypes[line[3]]

                    else:

                        # the first number has different dtype than the list

                        dtypes[name].append(

                            (mesh_names[0], ext + ply_dtypes[line[2]]))

                        # rest of the numbers have the same dtype

                        dt = ext + ply_dtypes[line[3]]



                    for j in range(1, 4):

                        dtypes[name].append((mesh_names[j], dt))

                else:

                    if fmt == "ascii":

                        dtypes[name].append(

                            (line[2].decode(), ply_dtypes[line[1]]))

                    else:

                        dtypes[name].append(

                            (line[2].decode(), ext + ply_dtypes[line[1]]))

            count += 1



        # for bin

        end_header = ply.tell()



    data = {}



    if fmt == 'ascii':

        top = count

        bottom = 0 if mesh_size is None else mesh_size



        names = [x[0] for x in dtypes["vertex"]]



        data["points"] = pd.read_csv(filename, sep=" ", header=None, engine="python",

                                     skiprows=top, skipfooter=bottom, usecols=names, names=names)



        for n, col in enumerate(data["points"].columns):

            data["points"][col] = data["points"][col].astype(

                dtypes["vertex"][n][1])



        if mesh_size is not None:

            

            top = count + points_size



            names = [x[0] for x in dtypes["face"]][1:]

            usecols = [1, 2, 3]



            data["mesh"] = pd.read_csv(

                filename, sep=" ", header=None, engine="python", skiprows=top, usecols=usecols, names=names)



            for n, col in enumerate(data["mesh"].columns):

                data["mesh"][col] = data["mesh"][col].astype(

                    dtypes["face"][n + 1][1])



    else:

        with open(filename, 'rb') as ply:

            ply.seek(end_header)

            points_np = np.fromfile(ply, dtype=dtypes["vertex"], count=points_size)

            if ext != sys_byteorder:

                points_np = points_np.byteswap().newbyteorder()

            data["points"] = pd.DataFrame(points_np)

            if (mesh_size is not None) and False:

                print('reading mesh', mesh_size)

                mesh_np = np.fromfile(ply, dtype=dtypes["face"], count=mesh_size)

                print('mesh', mesh_np.shape)

                if ext != sys_byteorder:

                    mesh_np = mesh_np.byteswap().newbyteorder()

                data["mesh"] = pd.DataFrame(mesh_np)

                data["mesh"].drop('n_points', axis=1, inplace=True)



    return data

def read_ply_points(c_file, **kwargs):

    out_df = read_ply(c_file)['points']

    for k,v in kwargs.items():

        out_df[k] = v

    return out_df
ply_path = glob(os.path.join(DATA_DIR, '*.ply'))[0]

png_path = glob(os.path.join(DATA_DIR, '*.png'))[0]

ply_data = read_ply_points(ply_path)

img_data = imread(png_path)



print(ply_data.keys(), img_data.shape)

plt.imshow(img_data)

ply_data.sample(4)
ply_data.sample(5000).plot.scatter('x', 'y')
ply_data.sample(5000).plot.scatter('x', 'z')
temp_df = ply_data.sample(1000)

fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))

ax1.quiver(temp_df['x'], temp_df['y'], temp_df['nx'], temp_df['ny'])
from mpl_toolkits.mplot3d import axes3d

temp_df = ply_data.sample(5000)

fig = plt.figure(figsize = (10, 10))

ax = fig.gca(projection='3d')

ax.quiver(temp_df['x'], temp_df['y'], temp_df['z'], 

          temp_df['nx'], temp_df['ny'], temp_df['nz'],

         length=0.1,

         normalize=True)

ax.view_init(-45, 30)
RAW_DIR = '../input/d45cd2dd-beaf-22aa-8529-9777b10721f4/'

!cat {os.path.join(RAW_DIR, 'depth00', 'camera.yaml')}
# decompress depth images and read them in

!unzip -qq {os.path.join(RAW_DIR, 'depth00', 'depth_images.zip')}

all_depth_maps = {int(os.path.splitext(k)[0].split('_')[-1]): imread(k) for k in glob('*.pnm')}

print('Map Shape', next(iter(all_depth_maps.values())).shape, 'Maps', len(all_depth_maps))

!rm *.pnm
fig, m_axs = plt.subplots(3, 3, figsize = (15, 15))

for (k, v), c_ax in zip(all_depth_maps.items(), m_axs.flatten()):

    c_ax.imshow(v)

    c_ax.set_title('Timestamp: {}'.format(k))

    c_ax.axis('off')
print('Camera Info')

!cat {os.path.join(RAW_DIR, 'color00', 'camera.yaml')}

print('\nTime Data')

time_df = pd.read_table(os.path.join(RAW_DIR, 'color00', 'timestamps.txt'), header=None, names = ['Timestamp'])

time_df['Frame'] = time_df.index

print(time_df.shape[0])

time_df.sample(3)
import cv2 # use opencv

def read_video_segment(in_path, vid_seg = None, seg_frames = 4):

    cap = cv2.VideoCapture(in_path)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    frames = []

    if cap.isOpened() and video_length > 0:

        frame_ids = [0]

        if seg_frames<0:

            frame_ids = range(video_length+1)

        else:

            if vid_seg is None:

                vid_seg = np.linspace(0, 1, seg_frames)

            else:

                vid_seg = np.clip(vid_seg, 0, 1)



            frame_ids = np.clip(video_length*vid_seg, 0, video_length-1).astype(int)

        count = 0

        success, image = cap.read()

        print('Loaded', video_length, 'frames video', image.shape, 'resolution.', 'Extracting', len(frame_ids), 'frames')

        while success:

            if count in frame_ids:

                frames.append(image)

            success, image = cap.read()

            count += 1

    return frames

color_frames = read_video_segment(os.path.join(RAW_DIR, 'color00', 'color_images.mp4'), seg_frames = -1)
fig, m_axs = plt.subplots(3, 3, figsize = (15, 15))

for v, c_ax in zip(color_frames, m_axs.flatten()):

    c_ax.imshow(v[:, :, ::-1]) # opencv uses BGR

    c_ax.axis('off')
time_df['color'] = time_df['Frame'].map(lambda i: color_frames[i])

depth_df = pd.DataFrame([{'Timestamp': k, 'depth': v} for k,v in all_depth_maps.items()])
combined_df = pd.merge(time_df, depth_df, how='outer', on='Timestamp').sort_values('Timestamp').drop('Frame', 1)

combined_df.sample(3)
combined_df['back_depth'] = combined_df['depth'].fillna(method='bfill')

combined_df['forward_depth'] = combined_df['depth'].fillna(method='ffill')

combined_df['avg_depth'] = combined_df.apply(lambda x: 0.5*x['back_depth']+0.5*x['forward_depth'], 1) # interpolate depth maps

clean_df = combined_df[['Timestamp', 'color', 'avg_depth', 'back_depth']].dropna()

print(clean_df.shape[0], 'completed frames')

clean_df.sample(3)
fig, m_axs = plt.subplots(3, 5, figsize = (24, 15))

for (_, c_row), (c_ax, d_ax, f_ax) in zip(clean_df.sample(9).iterrows(), m_axs.T):

    depth = c_row['avg_depth']

    c_img = c_row['color'][:, :, ::-1].swapaxes(0, 1)

    c_ax.imshow(c_img) # opencv uses BGR

    c_ax.set_title('{Timestamp}'.format(**c_row))

    c_ax.axis('off')

    d_ax.imshow(depth)

    d_ax.set_aspect(3)

    d_ax.set_title('Average Depth')

    

    f_ax.imshow(c_row['back_depth'])

    f_ax.set_aspect(3)

    f_ax.set_title('Last Depth')
from skimage.transform import resize as imresize

from skimage.filters import gaussian

_, test_row = next(clean_df.sample(1, random_state = 2015).iterrows())

depth = test_row['avg_depth']

c_img = test_row['color'][:, :, ::-1].swapaxes(0, 1)

print(depth.shape, c_img.shape)

raw_shape = (c_img.shape[0]//3, c_img.shape[1]//3)

n_color_img = imresize(c_img, raw_shape+(3,), order = 2, mode = 'reflect')

def process_depth(in_img):

    return np.sqrt(gaussian(imresize(in_img, raw_shape, order = 2, mode = 'reflect'), 4))



n_depth_img = process_depth(depth)

l_depth_img = process_depth(test_row['back_depth'])



print(n_color_img.shape, n_depth_img.shape)

fig, (c_ax, d_ax, f_ax) = plt.subplots(1, 3, figsize = (20, 4))

c_ax.imshow(c_img) # opencv uses BGR

c_ax.set_title('{Timestamp}'.format(**test_row))

c_ax.axis('off')

d_ax.imshow(n_depth_img)

d_ax.set_title('Average Depth')



f_ax.imshow(l_depth_img)

f_ax.set_title('Last Depth')
fig = plt.figure(figsize = (10, 10))

ax = fig.gca(projection='3d')

xx,yy = np.meshgrid(range(raw_shape[1]),range(raw_shape[0]))

ax.plot_surface(xx,yy,n_depth_img, 

                rstride=1, cstride=1, 

                facecolors=n_color_img.astype(np.float32),

                linewidth=0, antialiased=True)

ax.view_init(90, 90)
ax.view_init(0, 90)

fig
ax.view_init(-45, 90)

fig