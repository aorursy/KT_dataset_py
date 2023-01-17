URL = ['https://images.unsplash.com/photo-1557912407-eb2900cf49e8?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1350&q=80']

CONFIG = {}

CONFIG['MAX_DIM'] = [500]
import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib as mpl

import IPython.display as display

import PIL.Image

from tensorflow.keras.preprocessing import image

from itertools import product

import time



#### Functions



def my_product(inp):

    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))

            

# Download an image and read it into a NumPy array.

def download(url, max_dim=None):

  name = url.split('/')[-1]

  image_path = tf.keras.utils.get_file(name, origin=url)

  img = PIL.Image.open(image_path)

  if max_dim:

    img.thumbnail((max_dim, max_dim))

  return np.array(img)



# Normalize an image

def deprocess(img):

  img = 255*(img + 1.0)/2.0

  return tf.cast(img, tf.uint8)



# Display an image

def show(img):

  display.display(PIL.Image.fromarray(np.array(img)))



def calc_loss(img, model):

  # Pass forward the image through the model to retrieve the activations.

  # Converts the image into a batch of size 1.

  img_batch = tf.expand_dims(img, axis=0)

  layer_activations = model(img_batch)

  if len(layer_activations) == 1:

    layer_activations = [layer_activations]



  losses = []

  for act in layer_activations:

    loss = tf.math.reduce_mean(act)

    losses.append(loss)



  return  tf.reduce_sum(losses)



class DeepDream(tf.Module):

  def __init__(self, model):

    self.model = model



  @tf.function(

      input_signature=(

        tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),

        tf.TensorSpec(shape=[], dtype=tf.int32),

        tf.TensorSpec(shape=[], dtype=tf.float32),)

  )

  def __call__(self, img, steps, step_size):

      print("Tracing")

      loss = tf.constant(0.0)

      for n in tf.range(steps):

        with tf.GradientTape() as tape:

          # This needs gradients relative to `img`

          # `GradientTape` only watches `tf.Variable`s by default

          tape.watch(img)

          loss = calc_loss(img, self.model)



        # Calculate the gradient of the loss with respect to the pixels of the input image.

        gradients = tape.gradient(loss, img)



        # Normalize the gradients.

        gradients /= tf.math.reduce_std(gradients) + 1e-8 

        

        # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.

        # You can update the image by directly adding the gradients (because they're the same shape!)

        img = img + gradients*step_size

        img = tf.clip_by_value(img, -1, 1)



      return loss, img



def run_deep_dream_simple(img, steps=100, step_size=0.01):

  # Convert from uint8 to the range expected by the model.

  img = tf.keras.applications.inception_v3.preprocess_input(img)

  img = tf.convert_to_tensor(img)

  step_size = tf.convert_to_tensor(step_size)

  steps_remaining = steps

  step = 0

  while steps_remaining:

    if steps_remaining>100:

      run_steps = tf.constant(100)

    else:

      run_steps = tf.constant(steps_remaining)

    steps_remaining -= run_steps

    step += run_steps



    loss, img = deepdream(img, run_steps, tf.constant(step_size))

    

    display.clear_output(wait=True)

    show(deprocess(img))

    print ("Step {}, loss {}".format(step, loss))





  result = deprocess(img)

  display.clear_output(wait=True)

  show(result)



  return result



def random_roll(img, maxroll):

  # Randomly shift the image to avoid tiled boundaries.

  shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)

  img_rolled = tf.roll(img, shift=shift, axis=[0,1])

  return shift, img_rolled



class TiledGradients(tf.Module):

  def __init__(self, model):

    self.model = model



  @tf.function(

      input_signature=(

        tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),

        tf.TensorSpec(shape=[], dtype=tf.int32),)

  )

  def __call__(self, img, tile_size=512):

    shift, img_rolled = random_roll(img, tile_size)



    # Initialize the image gradients to zero.

    gradients = tf.zeros_like(img_rolled)

    

    # Skip the last tile, unless there's only one tile.

    xs = tf.range(0, img_rolled.shape[0], tile_size)[:-1]

    if not tf.cast(len(xs), bool):

      xs = tf.constant([0])

    ys = tf.range(0, img_rolled.shape[1], tile_size)[:-1]

    if not tf.cast(len(ys), bool):

      ys = tf.constant([0])



    for x in xs:

      for y in ys:

        # Calculate the gradients for this tile.

        with tf.GradientTape() as tape:

          # This needs gradients relative to `img_rolled`.

          # `GradientTape` only watches `tf.Variable`s by default.

          tape.watch(img_rolled)



          # Extract a tile out of the image.

          img_tile = img_rolled[x:x+tile_size, y:y+tile_size]

          loss = calc_loss(img_tile, self.model)



        # Update the image gradients for this tile.

        gradients = gradients + tape.gradient(loss, img_rolled)



    # Undo the random shift applied to the image and its gradients.

    gradients = tf.roll(gradients, shift=-shift, axis=[0,1])



    # Normalize the gradients.

    gradients /= tf.math.reduce_std(gradients) + 1e-8 



    return gradients 



def run_deep_dream_with_octaves(img, steps_per_octave=100, step_size=0.01, 

                                octaves=range(-2,3), octave_scale=1.3):

  base_shape = tf.shape(img)

  img = tf.keras.preprocessing.image.img_to_array(img)

  img = tf.keras.applications.inception_v3.preprocess_input(img)



  initial_shape = img.shape[:-1]

  img = tf.image.resize(img, initial_shape)

  for octave in octaves:

    # Scale the image based on the octave

    new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32)*(octave_scale**octave)

    img = tf.image.resize(img, tf.cast(new_size, tf.int32))



    for step in range(steps_per_octave):

      gradients = get_tiled_gradients(img)

      img = img + gradients*step_size

      img = tf.clip_by_value(img, -1, 1)



#       if step % 10 == 0:

#         display.clear_output(wait=True)

#         show(deprocess(img))

#         print ("Octave {}, Step {}".format(octave, step))

    

  result = deprocess(img)

  return result





def RunDeepDreamInstance(i, param):



    

    

    return img
# Show Original Image

original_img = download(URL[0], max_dim=CONFIG['MAX_DIM'][0])

show(original_img)

display.display(display.HTML('Image from unsplash.com by: <a "https://unsplash.com/@miqul">Michal Mrozek</a>'))
CONFIG = {

    'STEP_SIZE'     : [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1],

    'OCTAVE_STEPS'  : [100],

    'OCTAVE_SCALE'  : [1.2],

    'OCTAVE_RANGE'  : [range(-2,3)],

    'LAYERS'        : [['mixed5','mixed6'],

                       ['mixed1', 'mixed4', 'mixed6', 'mixed9']

                      ]

}
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')



# Iterating Over Multiple Configurations

print ('RUNNING ITERATIONS FOR {} COMBINATIONS...'.format(len([x for x in my_product(CONFIG)])))

param_df = pd.DataFrame()

for i, param in enumerate(my_product(CONFIG)):

    

    start_time = time.time()

    layers = [base_model.get_layer(name).output for name in param['LAYERS']]

    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)



    # Gradients

    get_tiled_gradients = TiledGradients(dream_model)



    # Maximize Loss

    img = run_deep_dream_with_octaves(img              = original_img, 

                                      steps_per_octave = param['OCTAVE_STEPS'], 

                                      step_size        = param['STEP_SIZE'], 

                                      octaves          = param['OCTAVE_RANGE'], 

                                      octave_scale     = param['OCTAVE_SCALE'])

    base_shape = tf.shape(img)[:-1]

    # display.clear_output(wait=True)

    img = tf.image.resize(img, base_shape)

    img = tf.image.convert_image_dtype(img/255.0, dtype=tf.uint8)

        

    image.save_img('STEP_SIZE-param_{}.jpg'.format(i+1),img)

    show(img)

    end_time = time.time()

    difference = round(end_time - start_time, 2)

    

    param_df = param_df.append(pd.DataFrame({

        'param_num': [i+1],

        'param'    : [param],

        'exec_time_sec': [difference]

    }))

    param_df.to_pickle('STEP_SIZE.pkl')

    

    

    print ('-'*50)

    print ('PARAM #', i+1, ':', param)

    print ('Completed in {} seconds'.format(difference))

    print ('-'*50)
CONFIG = {

    'STEP_SIZE'     : [0.01],

    'OCTAVE_STEPS'  : [10, 50, 100, 150, 200, 250, 300, 400],

    'OCTAVE_SCALE'  : [1.2],

    'OCTAVE_RANGE'  : [range(-2,3)],

    'LAYERS'        : [['mixed5','mixed6'],

                       ['mixed1', 'mixed4', 'mixed6', 'mixed9']

                      ],

    'MAX_DIM'       : [500],

}
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')



# Iterating Over Multiple Configurations

print ('RUNNING ITERATIONS FOR {} COMBINATIONS...'.format(len([x for x in my_product(CONFIG)])))

param_df = pd.DataFrame()

for i, param in enumerate(my_product(CONFIG)):

    

    start_time = time.time()

    layers = [base_model.get_layer(name).output for name in param['LAYERS']]

    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)



    # Gradients

    get_tiled_gradients = TiledGradients(dream_model)



    # Maximize Loss

    img = run_deep_dream_with_octaves(img              = original_img, 

                                      steps_per_octave = param['OCTAVE_STEPS'], 

                                      step_size        = param['STEP_SIZE'], 

                                      octaves          = param['OCTAVE_RANGE'], 

                                      octave_scale     = param['OCTAVE_SCALE'])

    base_shape = tf.shape(img)[:-1]

    # display.clear_output(wait=True)

    img = tf.image.resize(img, base_shape)

    img = tf.image.convert_image_dtype(img/255.0, dtype=tf.uint8)

    

    

    

    image.save_img('OCTAVE_STEPS-param_{}.jpg'.format(i+1),img)

    show(img)

    end_time = time.time()

    difference = round(end_time - start_time, 2)

    

    param_df = param_df.append(pd.DataFrame({

        'param_num': [i+1],

        'param'    : [param],

        'exec_time_sec': [difference]

    }))

    

    print ('-'*50)

    print ('PARAM #', i+1, ':', param)

    print ('Completed in {} seconds'.format(difference))

    print ('-'*50)

    

param_df.to_pickle('OCTAVE_STEPS.pkl')
CONFIG = {

    'STEP_SIZE'     : [0.01],

    'OCTAVE_STEPS'  : [100],

    'OCTAVE_SCALE'  : [0.5, 1, 1.1, 1.25, 1.5, 2],

    'OCTAVE_RANGE'  : [range(-2,3)],

    'LAYERS'        : [['mixed5','mixed6'],

                       ['mixed1', 'mixed4', 'mixed6', 'mixed9']

                      ],

    'MAX_DIM'       : [500],

}
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')



# Iterating Over Multiple Configurations

print ('RUNNING ITERATIONS FOR {} COMBINATIONS...'.format(len([x for x in my_product(CONFIG)])))

param_df = pd.DataFrame()

for i, param in enumerate(my_product(CONFIG)):

    

    start_time = time.time()

    layers = [base_model.get_layer(name).output for name in param['LAYERS']]

    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)



    # Gradients

    get_tiled_gradients = TiledGradients(dream_model)



    # Maximize Loss

    img = run_deep_dream_with_octaves(img              = original_img, 

                                      steps_per_octave = param['OCTAVE_STEPS'], 

                                      step_size        = param['STEP_SIZE'], 

                                      octaves          = param['OCTAVE_RANGE'], 

                                      octave_scale     = param['OCTAVE_SCALE'])

    base_shape = tf.shape(img)[:-1]

    # display.clear_output(wait=True)

    img = tf.image.resize(img, base_shape)

    img = tf.image.convert_image_dtype(img/255.0, dtype=tf.uint8)

    

    

    

    image.save_img('OCTAVE_SCALE-param_{}.jpg'.format(i+1),img)

    show(img)

    end_time = time.time()

    difference = round(end_time - start_time, 2)

    

    param_df = param_df.append(pd.DataFrame({

        'param_num': [i+1],

        'param'    : [param],

        'exec_time_sec': [difference]

    }))

    

    print ('-'*50)

    print ('PARAM #', i+1, ':', param)

    print ('Completed in {} seconds'.format(difference))

    print ('-'*50)

    

param_df.to_pickle('OCTAVE_SCALE.pkl')
CONFIG = {

    'STEP_SIZE'     : [0.01],

    'OCTAVE_STEPS'  : [100],

    'OCTAVE_SCALE'  : [1.25],

    'OCTAVE_RANGE'  : [range(-1,1), range(-2, 1),  range(-3, 1), range(-4, 1), range(-5,1),

                       range(-1,2), range(-1, 3),  range(-1, 4), range(-1,5), range(-5,5)

                      ],

    'LAYERS'        : [['mixed5','mixed6'],

                       ['mixed1', 'mixed4', 'mixed6', 'mixed9']

                      ],

    'MAX_DIM'       : [500],

}
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')



# Iterating Over Multiple Configurations

print ('RUNNING ITERATIONS FOR {} COMBINATIONS...'.format(len([x for x in my_product(CONFIG)])))

param_df = pd.DataFrame()

for i, param in enumerate(my_product(CONFIG)):

    

    start_time = time.time()

    layers = [base_model.get_layer(name).output for name in param['LAYERS']]

    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)



    # Gradients

    get_tiled_gradients = TiledGradients(dream_model)



    # Maximize Loss

    img = run_deep_dream_with_octaves(img              = original_img, 

                                      steps_per_octave = param['OCTAVE_STEPS'], 

                                      step_size        = param['STEP_SIZE'], 

                                      octaves          = param['OCTAVE_RANGE'], 

                                      octave_scale     = param['OCTAVE_SCALE'])

    base_shape = tf.shape(img)[:-1]

    # display.clear_output(wait=True)

    img = tf.image.resize(img, base_shape)

    img = tf.image.convert_image_dtype(img/255.0, dtype=tf.uint8)

    

    image.save_img('OCTAVE_RANGE-param_{}.jpg'.format(i+1),img)

    show(img)

    end_time = time.time()

    difference = round(end_time - start_time, 2)

    

    param_df = param_df.append(pd.DataFrame({

        'param_num': [i+1],

        'param'    : [param],

        'exec_time_sec': [difference]

    }))

    

    print ('-'*50)

    print ('PARAM #', i+1, ':', param)

    print ('Completed in {} seconds'.format(difference))

    print ('-'*50)

    

param_df.to_pickle('OCTAVE_RANGE.pkl')
CONFIG = {

    'STEP_SIZE'     : [0.01],

    'OCTAVE_STEPS'  : [150],

    'OCTAVE_SCALE'  : [1.25],

    'OCTAVE_RANGE'  : [range(-5,5)

                      ],

    'LAYERS'        : [['mixed5','mixed6'],

                       ['mixed1', 'mixed4', 'mixed6', 'mixed9']

                      ],

    'MAX_DIM'       : [500],

}
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')



# Iterating Over Multiple Configurations

print ('RUNNING ITERATIONS FOR {} COMBINATIONS...'.format(len([x for x in my_product(CONFIG)])))

param_df = pd.DataFrame()

for i, param in enumerate(my_product(CONFIG)):

    

    start_time = time.time()

    layers = [base_model.get_layer(name).output for name in param['LAYERS']]

    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)



    # Gradients

    get_tiled_gradients = TiledGradients(dream_model)



    # Maximize Loss

    img = run_deep_dream_with_octaves(img              = original_img, 

                                      steps_per_octave = param['OCTAVE_STEPS'], 

                                      step_size        = param['STEP_SIZE'], 

                                      octaves          = param['OCTAVE_RANGE'], 

                                      octave_scale     = param['OCTAVE_SCALE'])

    base_shape = tf.shape(img)[:-1]

    # display.clear_output(wait=True)

    img = tf.image.resize(img, base_shape)

    img = tf.image.convert_image_dtype(img/255.0, dtype=tf.uint8)

    

    image.save_img('RANDOME-param_{}.jpg'.format(i+1),img)

    show(img)

    end_time = time.time()

    difference = round(end_time - start_time, 2)

    

    param_df = param_df.append(pd.DataFrame({

        'param_num': [i+1],

        'param'    : [param],

        'exec_time_sec': [difference]

    }))

    

    print ('-'*50)

    print ('PARAM #', i+1, ':', param)

    print ('Completed in {} seconds'.format(difference))

    print ('-'*50)

    

param_df.to_pickle('RANDOME.pkl')