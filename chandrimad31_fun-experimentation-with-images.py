# For running inference on the TF-Hub module

import tensorflow as tf

import tensorflow_hub as hub



# For downloading the image

import matplotlib.pyplot as plt

import tempfile

from six.moves.urllib.request import urlopen

from six import BytesIO



# For drawing onto the image

import numpy as np

from PIL import Image

from PIL import ImageColor

from PIL import ImageDraw

from PIL import ImageFont

from PIL import ImageOps



# For measuring the inference time

import time



# Print Tensorflow version

print(tf.__version__)



# Check available GPU devices

print("The following GPU devices are available: %s" % tf.test.gpu_device_name())
def display_image(image):

  fig = plt.figure(figsize=(20, 15))

  plt.grid(False)

  plt.imshow(image)





def download_and_resize_image(url, new_width=256, new_height=256,

                              display=False):

  _, filename = tempfile.mkstemp(suffix=".jpg")

  response = urlopen(url)

  image_data = response.read()

  image_data = BytesIO(image_data)

  pil_image = Image.open(image_data)

  pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)

  pil_image_rgb = pil_image.convert("RGB")

  pil_image_rgb.save(filename, format="JPEG", quality=90)

  print("Image downloaded to %s." % filename)

  if display:

    display_image(pil_image)

  return filename





def draw_bounding_box_on_image(image,

                               ymin,

                               xmin,

                               ymax,

                               xmax,

                               color,

                               font,

                               thickness=4,

                               display_str_list=()):

  """Adds a bounding box to an image."""

  draw = ImageDraw.Draw(image)

  im_width, im_height = image.size

  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,

                                ymin * im_height, ymax * im_height)

  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),

             (left, top)],

            width=thickness,

            fill=color)



  # If the total height of the display strings added to the top of the bounding

  # box exceeds the top of the image, stack the strings below the bounding box

  # instead of above.

  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]

  # Each display_str has a top and bottom margin of 0.05x.

  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)



  if top > total_display_str_height:

    text_bottom = top

  else:

    text_bottom = top + total_display_str_height

  # Reverse list and print from bottom to top.

  for display_str in display_str_list[::-1]:

    text_width, text_height = font.getsize(display_str)

    margin = np.ceil(0.05 * text_height)

    draw.rectangle([(left, text_bottom - text_height - 2 * margin),

                    (left + text_width, text_bottom)],

                   fill=color)

    draw.text((left + margin, text_bottom - text_height - margin),

              display_str,

              fill="black",

              font=font)

    text_bottom -= text_height - 2 * margin





def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):

  """Overlay labeled boxes on an image with formatted scores and label names."""

  colors = list(ImageColor.colormap.values())



  try:

    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",

                              25)

  except IOError:

    print("Font not found, using default font.")

    font = ImageFont.load_default()



  for i in range(min(boxes.shape[0], max_boxes)):

    if scores[i] >= min_score:

      ymin, xmin, ymax, xmax = tuple(boxes[i])

      display_str = "{}: {}%".format(class_names[i].decode("ascii"),

                                     int(100 * scores[i]))

      color = colors[hash(class_names[i]) % len(colors)]

      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")

      draw_bounding_box_on_image(

          image_pil,

          ymin,

          xmin,

          ymax,

          xmax,

          color,

          font,

          display_str_list=[display_str])

      np.copyto(image, np.array(image_pil))

  return image
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Drumheller_%26_the_Tyrell_Museum_%287897901734%29.jpg/1200px-Drumheller_%26_the_Tyrell_Museum_%287897901734%29.jpg"  

downloaded_image_path = download_and_resize_image(image_url, 1280, 856, True)
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1" 

detector = hub.load(module_handle).signatures['default']
def load_img(path):

  img = tf.io.read_file(path)

  img = tf.image.decode_jpeg(img, channels=3)

  return img
def run_detector(detector, path):

  img = load_img(path)



  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

  start_time = time.time()

  result = detector(converted_img)

  end_time = time.time()



  result = {key:value.numpy() for key,value in result.items()}



  print("Found %d objects." % len(result["detection_scores"]))

  print("Inference time: ", end_time-start_time)



  image_with_boxes = draw_boxes(

      img.numpy(), result["detection_boxes"],

      result["detection_class_entities"], result["detection_scores"])



  display_image(image_with_boxes)
run_detector(detector, downloaded_image_path)
image_urls = [

  

  "https://upload.wikimedia.org/wikipedia/commons/d/d9/Verkehrsunfall1.jpg",

  

  "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Verkehrsunfall_L261_04.JPG/1200px-Verkehrsunfall_L261_04.JPG",

  

  "https://upload.wikimedia.org/wikipedia/commons/5/5e/BA38_Crash.jpg",

  

  "https://s.abcnews.com/images/2020/180905_2020_abby1_hpMain_16x9_992.jpg"

  ]
# Head-On Collision

image_1 = download_and_resize_image(image_urls[0], 1280, 856, True)

run_detector(detector, image_1)
# Motor Vehicle Injury

image_2 = download_and_resize_image(image_urls[1], 1280, 856, True)

run_detector(detector, image_2)
# Airplane Crash

image_3 = download_and_resize_image(image_urls[2], 1280, 856, True)

run_detector(detector, image_3)
# Surveillance Camera Image

image_4 = download_and_resize_image(image_urls[3], 1280, 856, True)

run_detector(detector, image_4)
import functools

import os



from matplotlib import gridspec

import matplotlib.pylab as plt

import numpy as np

import tensorflow as tf

import tensorflow_hub as hub



print("TF Version: ", tf.__version__)

print("TF-Hub version: ", hub.__version__)

print("Eager mode enabled: ", tf.executing_eagerly())

print("GPU available: ", tf.test.is_gpu_available())
# Image loading and visualization functions  



def crop_center(image):

  """Returns a cropped square image."""

  shape = image.shape

  new_shape = min(shape[1], shape[2])

  offset_y = max(shape[1] - shape[2], 0) // 2

  offset_x = max(shape[2] - shape[1], 0) // 2

  image = tf.image.crop_to_bounding_box(

      image, offset_y, offset_x, new_shape, new_shape)

  return image



@functools.lru_cache(maxsize=None)

def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):

  """Loads and preprocesses images."""

  # Cache image file locally

  image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)

  # Load and convert to float32 numpy array, add batch dimension, and normalize to range {0, 1}

  img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]

  if img.max() > 1.0:

    img = img / 255.

  if len(img.shape) == 3:

    img = tf.stack([img, img, img], axis=-1)

  img = crop_center(img)

  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)

  return img



def show_n(images, titles=('',)):

  n = len(images)

  image_sizes = [image.shape[1] for image in images]

  w = (image_sizes[0] * 6) // 320

  plt.figure(figsize=(w  * n, w))

  gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)

  for i in range(n):

    plt.subplot(gs[i])

    plt.imshow(images[i][0], aspect='equal')

    plt.axis('off')

    plt.title(titles[i] if len(titles) > i else '')

  plt.show()
# Load images



content_image_url = 'https://upload.wikimedia.org/wikipedia/commons/3/3f/Lana_Turner_still.JPG'  

style_image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg'

output_image_size = 384  



# The content image size can be arbitrary

content_img_size = (output_image_size, output_image_size)

# The style prediction model was trained with image size 256 

style_img_size = (256, 256) 



content_image = load_image(content_image_url, content_img_size)

style_image = load_image(style_image_url, style_img_size)

style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

show_n([content_image, style_image], ['Content image', 'Style image'])
# Load TF-Hub module

hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'

hub_module = hub.load(hub_handle)
outputs = hub_module(content_image, style_image)

stylized_image = outputs[0]
# Stylize content image with given style image

outputs = hub_module(tf.constant(content_image), tf.constant(style_image))

stylized_image = outputs[0]
# Visualize input images and the generated stylized image

show_n([content_image, style_image, stylized_image], titles=['Original content image', 'Style image', 'Stylized image'])
# Load more images to continue fun-experimenting



content_urls = dict(

  sea_turtle='https://upload.wikimedia.org/wikipedia/commons/d/d7/Green_Sea_Turtle_grazing_seagrass.jpg',

  tuebingen='https://upload.wikimedia.org/wikipedia/commons/0/00/Tuebingen_Neckarfront.jpg',

  grace_hopper='https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg'

  )



style_urls = dict(

  kanagawa_great_wave='https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg',

  hubble_pillars_of_creation='https://upload.wikimedia.org/wikipedia/commons/6/68/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg',

  van_gogh_starry_night='https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg',

  turner_nantes='https://upload.wikimedia.org/wikipedia/commons/b/b7/JMW_Turner_-_Nantes_from_the_Ile_Feydeau.jpg'

)



content_image_size = 384

style_image_size = 256

content_images = {k: load_image(v, (content_image_size, content_image_size)) for k, v in content_urls.items()}

style_images = {k: load_image(v, (style_image_size, style_image_size)) for k, v in style_urls.items()}

style_images = {k: tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME') for k, style_image in style_images.items()}
def fun_experiment(cont_image, st_image):

    content_name = cont_image

    style_name = st_image  

    stylized_image = hub_module(tf.constant(content_images[content_name]),tf.constant(style_images[style_name]))[0]

    show_n([content_images[content_name], style_images[style_name], stylized_image], titles=['Original content image', 'Style image', 'Stylized image'])
# Experiment No. 1

fun_experiment('sea_turtle', 'turner_nantes')
# Experiment No. 2

fun_experiment('tuebingen', 'kanagawa_great_wave')
# Experiment No. 3

fun_experiment('grace_hopper', 'hubble_pillars_of_creation')
!pip install wget
import os

import time

from PIL import Image

import numpy as np

import tensorflow as tf

import tensorflow_hub as hub

import matplotlib.pyplot as plt

import wget

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
url = 'https://upload.wikimedia.org/wikipedia/commons/9/99/Mouth.jpg'

fname = wget.download(url)
# Declaring Constants

IMAGE_PATH = fname

SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
def preprocess_image(image_path):

  """ Loads image from path and preprocesses to make it model ready

      Args:

        image_path: Path to the image file

  """

  hr_image = tf.image.decode_image(tf.io.read_file(image_path))

  # If PNG, remove the alpha channel. The model only supports

  # images with 3 color channels.

  if hr_image.shape[-1] == 4:

    hr_image = hr_image[...,:-1]

  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4

  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])

  hr_image = tf.cast(hr_image, tf.float32)

  return tf.expand_dims(hr_image, 0)



def save_image(image, filename):

  """

    Saves unscaled Tensor Images.

    Args:

      image: 3D image tensor. [height, width, channels]

      filename: Name of the file to save to.

  """

  if not isinstance(image, Image.Image):

    image = tf.clip_by_value(image, 0, 255)

    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())

  image.save("%s.jpg" % filename)

  print("Saved as %s.jpg" % filename)
%matplotlib inline

def plot_image(image, title=""):

  """

    Plots images from image tensors.

    Args:

      image: 3D image tensor. [height, width, channels].

      title: Title to display in the plot.

  """

  image = np.asarray(image)

  image = tf.clip_by_value(image, 0, 255)

  image = Image.fromarray(tf.cast(image, tf.uint8).numpy())

  plt.imshow(image)

  plt.axis("off")

  plt.title(title)
hr_image = preprocess_image(IMAGE_PATH)
# Plotting Original Resolution image

plot_image(tf.squeeze(hr_image), title="Original Image")

save_image(tf.squeeze(hr_image), filename="Original Image")
model = hub.load(SAVED_MODEL_PATH)
start = time.time()

fake_image = model(hr_image)

fake_image = tf.squeeze(fake_image)

print("Time Taken: %f" % (time.time() - start))
# Plotting Super Resolution Image

plot_image(tf.squeeze(fake_image), title="Enhanced Resolution")

save_image(tf.squeeze(fake_image), filename="Enhanced Resolution")