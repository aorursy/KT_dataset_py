import numpy as np
import math
import time
from tensorflow import keras
from pathlib import Path
from os import path
from keras.preprocessing.image import load_img
import IPython.display as display
# Try different sizes, e.g. 3 or 7.
CONV_SIZE=5
learning_rate=0.003
IMG_ROWS, IMG_COLS = CONV_SIZE, CONV_SIZE # input image dimensions
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)
model_name="SingleC2D"
model_plot_filename="C2D.png"
show_shapes=True
def transform_weights(wts):
    main_list=[]
    for wtss in wts:
        sub_list=[]
        for wtsss in wtss:
            sub_list.append(wtsss[0][0])
            
        main_list.append(sub_list)
    return np.asarray(main_list)
def create_model():
    sub_image = keras.layers.Input(shape=INPUT_SHAPE, name="image")
    c2d=keras.layers.Conv2D(1, (CONV_SIZE, CONV_SIZE), strides=math.ceil(CONV_SIZE/2), use_bias=False, name="C2D")(sub_image)
    model = keras.models.Model(inputs=sub_image, outputs=c2d, name=model_name)   
    return model
def animate(steps):
  if steps > 0:
      history = model.fit(inputs, outputs, epochs=1, verbose=False)
      loss = history.history['loss'][-1]
      out = model(inputs).numpy()[0][0][0][0]
      #if steps % 5 == 0 :
      #  print('[{:3d} {:5.2f} {:4.2f}]'.format(steps, loss, out), end=" ")
      wts = model.get_weights()
      wtsz = transform_weights(wts[0])
      plot[0].remove()      
      surface = get_plot_item(ax,x,y,wtsz)
      plot[0] = surface
  else:
      plot[0].remove()      
      surface = get_plot_item(ax,x,y,initial_wtsz)
      plot[0] = surface
  return (surface,)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
def display3dsubplots(z1, z2, title_text='Sub-image and Weights'):
  x = np.outer(np.linspace(0, CONV_SIZE-1, CONV_SIZE), np.ones(CONV_SIZE))
  y = x.copy().T

  spfig = make_subplots(
      rows=1, cols=2,
      specs=[[{'type': 'surface'}, {'type': 'surface'}]])
  spfig.add_trace(
      go.Surface(x=x, y=y, z=z1,
      showscale=False),
      row=1, col=1)

  spfig.add_trace(
      go.Surface(x=x, y=y, z=z2,
      showscale=False),
      row=1, col=2)
  spfig.update_layout(
      title_text=title_text,
      height=600,
      width=900)
  spf1 = go.FigureWidget(spfig)
  spf1.show()

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def get_plot_item(ax,x,y,z):
    #item = ax.plot_surface(x,y,z, cmap='viridis', edgecolor='none')
    item = ax.plot_wireframe(x, y, z, color='green')
    return item 

def mat3dplot(angle1, angle2):
  x = np.outer(np.linspace(0, CONV_SIZE-1, CONV_SIZE), np.ones(CONV_SIZE))
  y = x.copy().T
  fig = plt.figure(figsize=[12, 10])
  ax = plt.axes(projection='3d')
  ax.set_title('Weights')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  surface = get_plot_item(ax,x,y,initial_wtsz)
  ax.view_init(angle1, angle2)
  plot = [surface]
  return fig, plot, ax, x, y

model=create_model()
model.summary()
keras.utils.plot_model(model, model_plot_filename, show_shapes=show_shapes) 
wts = model.get_weights()
wtsz = transform_weights(wts[0])
initial_wtsz=wtsz
print(wtsz)
from keras.preprocessing.image import array_to_img
image_array1 = np.empty([3, 100, 50]) 
image_array1.fill(80) 
image_array2 = np.empty([3, 100, 50]) 
image_array2.fill(200) 
image_array = np.concatenate((image_array1, image_array2), axis=2)
img = array_to_img(image_array, data_format='channels_first', scale=False)
img=img.convert('L')
display.display(img)


from keras.preprocessing.image import img_to_array
left = 50-math.ceil(CONV_SIZE/2)
top = 50
right = left+CONV_SIZE
bottom = top+CONV_SIZE
sub_image = img.crop((left, top, right, bottom)) 
display.display(sub_image)
display.display(sub_image.resize((50,50)))
imageArray =np.asarray(sub_image)/255
print(imageArray)
display3dsubplots( imageArray, initial_wtsz, title_text='Sub-image and Initial Weights')
import tensorflow as tf
reference = CONV_SIZE*CONV_SIZE/2
batch_size = 500
X = tf.constant(np.array([imageArray]), dtype=tf.float32 )
Y = tf.constant(np.full((1,), reference), dtype=tf.float32 )
dataset = tf.data.Dataset.from_tensor_slices(( X , Y )) 
dataset = dataset.shuffle( 1 ).repeat( 1 ).batch( batch_size )
iterator = dataset.__iter__()
inputs , outputs = iterator.get_next()
print(inputs, outputs)
model.compile(keras.optimizers.SGD(learning_rate), loss='mse', run_eagerly=True)
fig, plot, ax, x, y = mat3dplot(65, 45)
import matplotlib
from matplotlib import animation
matplotlib.rcParams['animation.embed_limit'] = 2**128
print("Target output is", reference)
print('Initial output {:4.2f}'.format(model(inputs).numpy()[0][0][0][0]))
anim = animation.FuncAnimation(fig, animate, interval=50, frames=100, blit=True)

from IPython.display import HTML
HTML(anim.to_jshtml())
print('Final output {:4.2f}'.format(model(inputs).numpy()[0][0][0][0]))
print("Sub-image")
print(imageArray)
wts = model.get_weights()
wtsz = transform_weights(wts[0])
print("Final weights")
print(wtsz)
print("sum", np.sum(wtsz))
print("Initial weights")
print(initial_wtsz)
print("sum", np.sum(initial_wtsz))
display3dsubplots( imageArray, wtsz, title_text='Sub image and final Weights')
