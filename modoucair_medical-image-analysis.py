import pydicom

import tensorflow as tf

from tensorflow.keras import layers

import pandas as pd

import PIL

import plotly.graph_objects as go

import matplotlib.pyplot as plt

import numpy as np

import glob

import os

import cv2

from skimage import measure

import scipy

from plotly.tools import FigureFactory as FF

from plotly.graph_objs import *

from scipy.ndimage import zoom

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.express as px

def load_scan(path):

    slices = [pydicom.read_file(path+"/"+s) for s in os.listdir(path) ]

    slices.sort(key = lambda x: int(x.AcquisitionNumber))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    for s in slices:

        s.SliceThickness = slice_thickness

        s.SamplesPerPixel = 1

        

    return slices



def get_pixels_hu(scans):

    image = np.stack([s.pixel_array for s in scans[:100]])

    # Convert to int16 (from sometimes int16), 

    # should be possible as values should always be low enough (<32k)

    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1

    # The intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)

    intercept = scans[0].RescaleIntercept

    slope = scans[0].RescaleSlope

    

    if slope != 1:

        image = slope * image.astype(np.float64)

        image = image.astype(np.int16)

        

    image += np.int16(intercept)

    

    return np.array(image, dtype=np.int16)

def make_mesh(image,threshold=100):

    print( "Transposing surface")

    p = image.transpose(2,1,0)

    print( "Calculating surface")

    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold) 

    return verts, faces



def plotly_3d(verts, faces):

    x,y,z = zip(*verts)   

    print("Drawing")

    # Make the colormap single color since the axes are positional not intensity. 

    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']

    #colormap = ['rgb(100,149,237)','rgb(100,149,237)']

    #mesh.set_facecolor(face_color)

    fig = FF.create_trisurf(x=x,

                        y=y, 

                        z=z, 

                        plot_edges=False,

                        colormap=colormap,

                        simplices=faces,

                        backgroundcolor='rgb(64, 64, 64)',

                        title="Interactive Visualization")

    iplot(fig)

def get_y(df):

    dic = {True:1,False:0}

    df['Contrast'] = df['Contrast'].map(dic)

    y =df['Contrast'].values

    return y



path =  "/kaggle/input/siim-medical-images/dicom_dir/"

#id=0

patient = load_scan(path)

imgs = get_pixels_hu(patient)
fig = plt.figure(figsize=(20,20))

for num,image in enumerate(imgs[:12]):

    ax = fig.add_subplot(3,4,num+1)

    ax.imshow(image, cmap=plt.cm.bone)

    ax.set_title(f"The age of this patient:{patient[num].PatientAge}\nAnd is a {patient[num].PatientSex}")

plt.show()
img = np.copy(imgs[0])

fig = px.histogram(x=img.flatten())

fig.show()
seg1 = (img<-2000)

seg2 = (img>-2000) & (img<-1000)

seg3 = (img>-1000) & (img<-500)

seg4 = (img>-500)

all_seg = np.zeros((img.shape[0],img.shape[1],3))

all_seg[seg1] = (1,0,0)

all_seg[seg2] = (0,1,0)

all_seg[seg3] = (0,0,1)

all_seg[seg4] = (1,1,0)

#plt.imshow(all_seg)

#plt.show()

fig = px.imshow(all_seg)

fig.show()
all_seg[seg1] = (1,1,1)

all_seg[seg2] = (1,1,1)

all_seg[seg3] = (0,0,1)

all_seg[seg4] = (1,1,1)

kernel = np.ones((2,2),np.uint8)

erosion = cv2.erode(all_seg,kernel,iterations = 1)

dilation = cv2.dilate(all_seg,kernel,iterations = 1)

fig = px.imshow(all_seg,color_continuous_scale='gray')

fig.show()
img = cv2.rectangle(all_seg,(50,80),(446,389),(0,0,255),2)

plt.imshow(img,cmap=plt.cm.bone)

plt.show()
img1 = np.copy(imgs[0])

img1[img1>=-500] = 255

img1[img1<=-1000]=255

kernel = np.ones((2,2),np.uint16)

erosion = cv2.erode(img1,kernel,iterations = 2)

plt.imshow(erosion,cmap=plt.cm.gray)

plt.show()
edged=cv2.Canny(erosion.astype(np.uint8),30,200)

plt.imshow(edged,plt.cm.bone)

plt.show()
kernel = np.ones((5,5),np.uint8)

x,y,w,h =  50,80,446,389

ROI = erosion[y:h, x:w]

#plt.imshow(ROI,plt.cm.bone)

# Iterate thorugh contours and filter for ROI

fig = px.imshow(ROI,color_continuous_scale="gray")

fig.show()
zoomed = zoom(imgs.astype(np.float32), 0.25)

v, f = make_mesh(zoomed,threshold=-350)

plotly_3d(v, f)

volume=zoomed
r, c = volume[6].shape

# Define frames

import plotly.graph_objects as go

nb_frames = 25

fig = go.Figure(frames=[go.Frame(data=go.Surface(

    z=(6.7 - k * 0.1) * np.ones((r, c)),

    surfacecolor=np.flipud(volume[24 - k]),

    cmin=0, cmax=200

    ),

    name=str(k) 

    )

    for k in range(nb_frames)])



# Add data to be displayed before animation starts

fig.add_trace(go.Surface(

    z=6.7 * np.ones((r, c)),

    surfacecolor=np.flipud(volume[24]),

    colorscale="gray",

    cmin=0, cmax=200,

    colorbar=dict(thickness=20, ticklen=4)

    ))





def frame_args(duration):

    return {

            "frame": {"duration": 1500},

            "mode": "immediate",

            "fromcurrent": True,

            "transition": {"duration": 1500, "easing": "linear"},

        }



sliders = [

            {

                "pad": {"b": 10, "t": 60},

                "len": 0.9,

                "x": 0.1,

                "y": 0,

                "steps": [

                    {

                        "args": [[f.name], frame_args(0)],

                        "label": str(k),

                        "method": "animate",

                    }

                    for k, f in enumerate(fig.frames)

                ],

            }

        ]



fig.update_layout(

         title='Slices in volumetric ',

         width=600,

         height=600,

         scene=dict(

                    zaxis=dict(range=[-0.1, 6.8], autorange=False),

                    aspectratio=dict(x=1, y=1, z=1),

                    ),

         updatemenus = [

            {

                "buttons": [

                    {

                        "args": [None, frame_args(50)],

                        "label": "&#9654;", 

                        "method": "animate",

                    },

                    {

                        "args": [[None], frame_args(0)],

                        "label": "&#9724;", # pause symbol

                        "method": "animate",

                    },

                ],

                "direction": "left",

                "pad": {"r": 10, "t": 70},

                "type": "buttons",

                "x": 0.1,

                "y": 0,

            }

         ],

         sliders=sliders

)

fig.show()
df = pd.read_csv('/kaggle/input/siim-medical-images/overview.csv')

df.head()
y = get_y(df)
model = tf.keras.models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(512, 512, 1)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(2))
model.summary()
X = imgs.reshape([-1,512, 512,1])

train_ds = tf.data.Dataset.from_tensor_slices(

    (X, y)).shuffle(10000).batch(100)

Xtrain,ytrain = [],[]

for i ,g in train_ds:

    Xtrain.append(i)

    ytrain.append(g)
model.compile(optimizer='adam',

              loss=tf.keras.losses.sparse_categorical_crossentropy,

              metrics=['accuracy'])



history = model.fit(Xtrain,ytrain, epochs=10, 

                    validation_split=0.1)
fig = go.Figure()

fig.add_trace(go.Scatter(x=history.epoch, y=history.history['accuracy'],line_color='rgb(231,107,243)',

    name='Accuracy',fill='tonexty'))

fig.add_trace(go.Scatter(x=history.epoch, y=history.history['val_accuracy'],line_color='yellow',opacity=0.1,

    name='val_accuracy',fill='tozeroy'))

fig.update_layout(

Layout(

    paper_bgcolor='black',

    plot_bgcolor='black'),title_text=' Ops! The Sise of the data is so tiny  ')



fig.update_traces(mode='lines')

fig.show()