# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os

from pprint import pprint

from six import BytesIO

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow_hub as hub

from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps

from tqdm import tqdm
def format_prediction_string(image_id, result):

    prediction_strings = []

    

    for i in range(len(result['detection_scores'])):

        class_name = result['detection_class_names'][i].decode("utf-8")

        boxes = result['detection_boxes'][i]

        score = result['detection_scores'][i]

        

        prediction_strings.append(

            f"{class_name} {score} " + " ".join(map(str, boxes))

        )

        

    prediction_string = " ".join(prediction_strings)



    return {

        "ImageID": image_id,

        "PredictionString": prediction_string

    }
def display_image(image):

    fig = plt.figure(figsize=(20, 15))

    plt.grid(False)

    plt.axis('off')

    plt.imshow(image)
def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):

    """Overlay labeled boxes on an image with formatted scores and label names."""

    colors = list(ImageColor.colormap.values())



    try:

        font = ImageFont.truetype(

            "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",

            25)

    except IOError:

        print("Font not found, using default font.")

        font = ImageFont.load_default()



    for i in range(min(boxes.shape[0], max_boxes)):

        if scores[i] >= min_score:

            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())

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
import glob



INPUT_PATH_PNG = "../input/aptos2019-blindness-detection/train_images/"

files_png_init = sorted(glob.glob(INPUT_PATH_PNG + '*.png'))

files_png_init = files_png_init[:300]

print('PNG Files: {}'.format(len(files_png_init)))



os.mkdir('/dev/shm/4/')

files_png = []

for f in files_png_init:

    new_path = '/dev/shm/4/' + os.path.basename(f)

    shutil.copy(f, new_path)

    files_png.append(new_path)



INPUT_PATH_JPG_SMALL = "../input/open-images-2019-object-detection/test/"

files_jpg_small_init = sorted(glob.glob(INPUT_PATH_JPG_SMALL + '*.jpg'))

files_jpg_small_init = files_jpg_small_init[:3000]

print('JPG small files: {}'.format(len(files_jpg_small_init)))



os.mkdir('/dev/shm/5/')

files_jpg_small = []

for f in files_jpg_small_init:

    new_path = '/dev/shm/5/' + os.path.basename(f)

    shutil.copy(f, new_path)

    files_jpg_small.append(new_path)



INPUT_PATH_JPG_BIG = "../input/sp-society-camera-model-identification/train/"

files_jpg_big_init = sorted(glob.glob(INPUT_PATH_JPG_BIG + '*/*.jpg'))

files_jpg_big_init = files_jpg_big_init[:300]

print('JPG big files: {}'.format(len(files_jpg_big_init)))



os.mkdir('/dev/shm/6/')

files_jpg_big = []

for f in files_jpg_big_init:

    new_path = '/dev/shm/6/' + os.path.basename(f)

    shutil.copy(f, new_path)

    files_jpg_big.append(new_path)
import time



start_time = time.time()

d = []

for f in files_jpg_small:

    a = jpeg.JPEG(f).decode()

    d.append(a)

print('Time to read {} JPEGs small for libjpeg-turbo (jpeg4py): {:.2f} sec'.format(len(files_jpg_small), time.time() - start_time))



start_time = time.time()

d = []

for f in files_jpg_big:

    a = jpeg.JPEG(f).decode()

    d.append(a)

print('Time to read {} JPEGs big for libjpeg-turbo (jpeg4py): {:.2f} sec'.format(len(files_jpg_big), time.time() - start_time))
start_time = time.time()

d = []

for f in files_jpg_small:

    b = cv2.imread(f)

    # b = np.transpose(b, (1, 0, 2))

    # b = np.flip(b, axis=0)

    b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)

    d.append(b)

print('Time to read {} JPEGs small for cv2 with BGR->RGB conversion: {:.2f} sec'.format(len(files_jpg_small), time.time() - start_time))



start_time = time.time()

d = []

for f in files_jpg_big:

    b = cv2.imread(f)

    # b = np.transpose(b, (1, 0, 2))

    # b = np.flip(b, axis=0)

    b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)

    d.append(b)

print('Time to read {} JPEGs big for cv2 with BGR->RGB conversion: {:.2f} sec'.format(len(files_jpg_big), time.time() - start_time))



start_time = time.time()

d = []

for f in files_png:

    b = cv2.imread(f)

    # b = np.transpose(b, (1, 0, 2))

    # b = np.flip(b, axis=0)

    b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)

    d.append(b)

print('Time to read {} PNGs for cv2 with BGR->RGB conversion: {:.2f} sec'.format(len(files_png), time.time() - start_time))
start_time = time.time()

d = []

for f in files_jpg_small:

    b = cv2.imread(f)

    d.append(b)

print('Time to read {} JPEGs small for cv2 no conversion: {:.2f} sec'.format(len(files_jpg_small), time.time() - start_time))



start_time = time.time()

d = []

for f in files_jpg_big:

    b = cv2.imread(f)

    d.append(b)

print('Time to read {} JPEGs big for cv2 no conversion: {:.2f} sec'.format(len(files_jpg_big), time.time() - start_time))



start_time = time.time()

d = []

for f in files_png:

    b = cv2.imread(f)

    d.append(b)

print('Time to read {} PNGs for cv2 no conversion: {:.2f} sec'.format(len(files_png), time.time() - start_time))
start_time = time.time()

d = []

for f in files_jpg_small:

    c = Image.open(f)

    c = np.array(c)

    d.append(c)

print('Time to read {} JPEGs small for PIL: {:.2f} sec'.format(len(files_jpg_small), time.time() - start_time))



start_time = time.time()

d = []

for f in files_jpg_big:

    c = Image.open(f)

    c = np.array(c)

    d.append(c)

print('Time to read {} JPEGs big for PIL: {:.2f} sec'.format(len(files_jpg_big), time.time() - start_time))



start_time = time.time()

d = []

for f in files_png:

    c = Image.open(f)

    c = np.array(c)

    d.append(c)

print('Time to read {} PNGs for PIL: {:.2f} sec'.format(len(files_png), time.time() - start_time))
start_time = time.time()

d = []

plugin = 'matplotlib'

for f in files_jpg_small:

    c = skimage.io.imread(f, plugin=plugin)

    c = np.array(c)

    d.append(c)

print('Time to read {} JPEGs small for skimage.io Plugin: {}: {:.2f} sec'.format(len(files_jpg_small), plugin, time.time() - start_time))



start_time = time.time()

d = []

plugin = 'matplotlib'

for f in files_jpg_big:

    c = skimage.io.imread(f, plugin=plugin)

    c = np.array(c)

    d.append(c)

print('Time to read {} JPEGs big for skimage.io Plugin: {}: {:.2f} sec'.format(len(files_jpg_big), plugin, time.time() - start_time))



start_time = time.time()

d = []

plugin = 'matplotlib'

for f in files_png:

    c = skimage.io.imread(f, plugin=plugin)

    c = np.array(c)

    d.append(c)

print('Time to read {} PNGs for skimage.io Plugin: {}: {:.2f} sec'.format(len(files_png), plugin, time.time() - start_time))
pd.read_csv("../input/sample_submission.csv").head()
start_time = time.time()

d = []

for f in files_jpg_small:

    c = imageio.imread(f)

    d.append(c)

print('Time to read {} JPEGs small for Imageio (no rotate): {:.2f} sec'.format(len(files_jpg_small), time.time() - start_time))



start_time = time.time()

d = []

for f in files_jpg_big:

    c = imageio.imread(f)

    d.append(c)

print('Time to read {} JPEGs big for Imageio (no rotate): {:.2f} sec'.format(len(files_jpg_big), time.time() - start_time))



start_time = time.time()

d = []

for f in files_png:

    c = imageio.imread(f)

    d.append(c)

print('Time to read {} PNGs for Imageio (no rotate): {:.2f} sec'.format(len(files_png), time.time() - start_time))
images = os.listdir("../input/test")

images[:100]
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import cv2





#read the first jpg file

img = cv2.imread('../input/test/b4c3b52a8723d431.jpg',0)

#img = cv2.imread('../input/test/b4c3b52a8723d431.jpg')



#check the array of the first jpg file

img
#view the array as an image

plt.imshow(img)
x= '../input/test/'

myList = [ x + i for i in images[:100]]
for i in myList:

    plt.imshow( cv2.imread(i) ) 

    plt.show()
image_filenames = os.listdir("../input/test/")



import random

for i in range(10):

    index = random.randrange(len(image_filenames))

    path = "../input/test/" + "/" + image_filenames[index]

    src_img = cv2.imread(path)

    fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

    plt.imshow(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))

    plt.show()
import tensorflow as tf

import tensorflow_hub as hub



import tempfile

from six.moves.urllib.request import urlopen

from six import BytesIO



from PIL import Image

from PIL import ImageColor

from PIL import ImageDraw

from PIL import ImageFont

from PIL import ImageOps



import time



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





def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=()):

    draw = ImageDraw.Draw(image)

    im_width, im_height = image.size

    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)

    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)



    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]

    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)



    if top > total_display_str_height:

        text_bottom = top

    else:

        text_bottom = bottom + total_display_str_height

    for display_str in display_str_list[::-1]:

        text_width, text_height = font.getsize(display_str)

        margin = np.ceil(0.05 * text_height)

        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)

        draw.text((left + margin, text_bottom - text_height - margin), display_str, fill="black", font=font)

        text_bottom -= text_height - 2 * margin





def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):

    colors = list(ImageColor.colormap.values())



    try:

        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",

                              25)

    except IOError:

        print("Font not found, using default font.")

        font = ImageFont.load_default()



    for i in range(min(boxes.shape[0], max_boxes)):

        if scores[i] >= min_score:

            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())

            display_str = "{}: {}%".format(class_names[i].decode("ascii"), int(100 * scores[i]))

            color = colors[hash(class_names[i]) % len(colors)]

            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")

            draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font, display_str_list=[display_str])

        np.copyto(image, np.array(image_pil))

    return image
image_url = "https://farm1.staticflickr.com/4032/4653948754_c0d768086b_o.jpg"  #@param

downloaded_image_path = download_and_resize_image(image_url, 1280, 856, True)
image_urls = ["https://farm7.staticflickr.com/8092/8592917784_4759d3088b_o.jpg",

              "https://farm6.staticflickr.com/2598/4138342721_06f6e177f3_o.jpg"]



for image_url in image_urls:

    image_path = download_and_resize_image(image_url, 640, 480)

    with tf.gfile.Open(image_path, "rb") as binfile:

        image_string = binfile.read()



    inference_start_time = time.clock()

    result_out, image_out = session.run([result, decoded_image], feed_dict={image_string_placeholder: image_string})

    print("Found %d objects." % len(result_out["detection_scores"]))

    print("Inference took %.2f seconds." % (time.clock()-inference_start_time))



    image_with_boxes = draw_boxes(

    np.array(image_out), result_out["detection_boxes"],

    result_out["detection_class_entities"], result_out["detection_scores"])



    display_image(image_with_boxes)
!pip install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.3/imageai-2.0.3-py3-none-any.whl
!wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5
!wget https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv
s_sub = pd.read_csv('../input/sample_submission.csv')

s_sub.head()
test_filename = os.listdir('../input/test')

test_filename[:5]

labelMap = pd.read_csv('class-descriptions-boxable.csv', header=None, names=['labelName', 'Label'])

labelMap.head()
# Show one image

def show_image_by_index(i):

    sample_image = plt.imread(f'../input/test/{test_filename[i]}')

    plt.imshow(sample_image)



def show_image_by_filename(filename):

    sample_image = plt.imread(filename)

    plt.imshow(sample_image)
show_image_by_index(22)
show_image_by_filename(f'../input/test/e7c0991d9a37bdef.jpg')
from imageai.Detection import ObjectDetection
#Get the path to the working directory

execution_path = os.getcwd()
%%time

# load model

detector = ObjectDetection()

detector.setModelTypeAsYOLOv3()

detector.setModelPath(os.path.join(execution_path, "yolo.h5"))

detector.loadModel()
%%time

# test detection on one image

detections = detector.detectObjectsFromImage(input_image=os.path.join('../input/test', 'e7c0991d9a37bdef.jpg'),

                                                                      #test_filename[64]), 

                                             output_image_path=os.path.join(execution_path , "result.jpg"),

#                                            output_type = 'array',

                                             extract_detected_objects = False)

for eachObject in detections:

    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )



# show the result

show_image_by_filename('./result.jpg')
#view detection variable

detections

def format_prediction_string(image_id, result, labelMap, xSize, ySize):

    prediction_strings = []

    #print(xSize, ySize)

    for i in range(len(result)):

        class_name = result[i]['name'].capitalize()

        class_name = pd.DataFrame(labelMap.loc[labelMap['Label'].isin([class_name])]['labelName'])

        #print(result[i]['box_points'])

        xMin = result[i]['box_points'][0] / xSize

        xMax = result[i]['box_points'][2] / xSize

        yMin = result[i]['box_points'][1] / ySize

        yMax = result[i]['box_points'][3] / ySize

        

        if len(class_name) > 0:

            class_name = class_name.iloc[0]['labelName']

            boxes = [xMin, yMin, xMax, yMax]#result[i]['box_points']

            score = result[i]['percentage_probability']



            prediction_strings.append(

                f"{class_name} {score} " + " ".join(map(str, boxes))

            )

        

    prediction_string = " ".join(prediction_strings)



    return {

            "ImageID": image_id,

            "PredictionString": prediction_string

            }
%%time

# Test prediction on input images

res = []

for i in tqdm(os.listdir('../input/test')[20:25]):

    detections = detector.detectObjectsFromImage(input_image=os.path.join('../input/test', i),

                                                 output_image_path=os.path.join(execution_path , "result.jpg"),

                                                 #output_type = 'array',

                                                 extract_detected_objects = False)

    currentImg = Image.open(os.path.join('../input/test', i))

    xSize = currentImg.size[0]

    ySize = currentImg.size[1]

    #print(xSize, ySize)

    p = format_prediction_string(i, detections, labelMap, xSize, ySize)

    res.append(p)
# Convert res variable to DataFrame

pred_df = pd.DataFrame(res)

pred_df.head()
# Get the file name without extension

pred_df['ImageID'] = pred_df['ImageID'].map(lambda x: x.split(".")[0])
pred_df.head()