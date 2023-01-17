import pandas as pd

import numpy as np



BUCKET = 'gs://<your-bucket-name>' # change this to actual storage bucket where images are stored



pickles='/kaggle/input/starter-arthropod-taxonomy-orders-data-exploring/'

labels=pd.read_pickle(pickles+'ArTaxOr_labels.pkl')

df=pd.read_pickle(pickles+'ArTaxOr_filelist.pkl')

anno=pd.read_pickle(pickles+'ArTaxOr_objects.pkl')
df = df.sample(frac=1).reset_index(drop=True) # shuffle dataset (not really required, AutoML will do)

automl=pd.DataFrame(columns=['set', 'file', 'label', 'xmin1', 'ymin1', 'xmax2', 'ymin2', 'xmax3', 'ymax3', 'xmin4' , 'ymax4'])

for i in range(len(df)):

    an=anno[anno.id == df.iloc[i].id]

    for j in range(len(an)):

        automl=automl.append({'set': 'UNASSIGNED',

                              'file': BUCKET+an.file.iloc[j].replace('/kaggle/input',''),

                              'label': an.label.iloc[j],

                              'xmin1': an.left.iloc[j],

                              'ymin1': an.top.iloc[j],

                              'xmax2': an.right.iloc[j],

                              'ymin2': an.top.iloc[j],

                              'xmax3': an.right.iloc[j],

                              'ymax3': an.bottom.iloc[j],

                              'xmin4': an.left.iloc[j],

                              'ymax4': an.bottom.iloc[j]}, ignore_index=True)

automl.to_csv('./ArTaxOr.csv', index=False, header=False)

!head -3 ./ArTaxOr.csv
import urllib.request



model_url='https://github.com/geddy11/ArTaxOr-models/raw/master/TensorFlow/AutoML/tflite_model-ArTaxOr1.0.0_dataset_20191023_model.tflite'

urllib.request.urlretrieve(model_url, 'automl_trained.tflite')
from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



!pip install python-resize-image

from PIL import Image, ImageFont, ImageDraw

from resizeimage import resizeimage

import glob, os.path

import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline
def attribution(file):

    with Image.open(file) as img:

        exif_data = img._getexif()

    s='Photo: unknown'

    if exif_data is not None:

        if 37510 in exif_data:

            if len(exif_data[37510]) > 0:

                s = exif_data[37510][8:].decode('ascii')

        if 315 in exif_data:

            if len(exif_data[315]) > 0:

                s = 'Photo: ' + exif_data[315]

    return s



def resize_image(file, width, height, stretch=False):

    with Image.open(file) as im:

        img = im.resize((width, height)) if stretch else resizeimage.resize_contain(im, [width, height])

    img=img.convert("RGB")    

    return img, attribution(file)



fontname = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

font = ImageFont.truetype(fontname, 20) if os.path.isfile(fontname) else ImageFont.load_default()



def bbox(img, xmin, ymin, xmax, ymax, color, width, label, score):

    draw = ImageDraw.Draw(img)

    xres, yres = img.size[0], img.size[1]

    box = np.multiply([xmin, ymin, xmax, ymax], [xres, yres, xres, yres]).astype(int).tolist()

    txt = " {}: {}%" if score >= 0. else " {}"

    txt = txt.format(label, round(score, 1))

    ts = draw.textsize(txt, font=font)

    draw.rectangle(box, outline=color, width=width)

    if len(label) > 0:

        if box[1] >= ts[1]+3:

            xsmin, ysmin = box[0], box[1]-ts[1]-3

            xsmax, ysmax = box[0]+ts[0]+2, box[1]

        else:

            xsmin, ysmin = box[0], box[3]

            xsmax, ysmax = box[0]+ts[0]+2, box[3]+ts[1]+1

        draw.rectangle([xsmin, ysmin, xsmax, ysmax], fill=color)

        draw.text((xsmin, ysmin), txt, font=font, fill='white')
def hex_to_rgb(value):

    value = value.lstrip('#')

    lv = len(value)

    return tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3))



def plot_img_pred(img, axes, scores, boxes, classes, title, by=''):

    for i in range(len(scores)):

        if scores[i]> 0.5 and classes[i]>0:

            label = labels.name.iloc[int(classes[i]-1)]

            color = hex_to_rgb(labels[labels.name == label].color.iloc[0])

            bbox(img, boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2], color, 2, label, 100*scores[i])

    plt.setp(axes, xticks=[], yticks=[])

    axes.set_title(title) if by == '' else axes.set_title(title+'\n'+by)

    plt.imshow(img)

    

def plot_img_gt(img, axes, boxes, stretch, title, by=''):

    wscale = 1. if stretch else min(1,boxes.xres.iloc[0]/boxes.yres.iloc[0])

    hscale = 1. if stretch else min(1,boxes.yres.iloc[0]/boxes.xres.iloc[0])

    for i in range(len(boxes)):

        label = boxes.label.iloc[i]

        color = hex_to_rgb(labels[labels.name == label].color.iloc[0])

        xmin = .5+(boxes.xcenter.iloc[i]-.5)*wscale-.5*wscale*boxes.width.iloc[i]

        ymin = .5+(boxes.ycenter.iloc[i]-.5)*hscale-.5*hscale*boxes.height.iloc[i]

        xmax = .5+(boxes.xcenter.iloc[i]-.5)*wscale+.5*wscale*boxes.width.iloc[i]

        ymax = .5+(boxes.ycenter.iloc[i]-.5)*hscale+.5*hscale*boxes.height.iloc[i]

        bbox(img, xmin, ymin, xmax, ymax, color, 2, label, -1)

    plt.setp(axes, xticks=[], yticks=[])

    axes.set_title(title) if by == '' else axes.set_title(title+'\n'+by)

    plt.imshow(img)
interpreter = tf.lite.Interpreter(model_path='automl_trained.tflite')

interpreter.allocate_tensors()

input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()



def predict(img):

    input_data = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])

    classes = interpreter.get_tensor(output_details[1]['index'])

    scores = interpreter.get_tensor(output_details[2]['index'])

    num = interpreter.get_tensor(output_details[3]['index'])

    return scores, classes, boxes, num
pickles='/kaggle/input/starter-arthropod-taxonomy-orders-testset/'

labels=pd.read_pickle(pickles+'testset_labels.pkl')

df=pd.read_pickle(pickles+'testset_filelist.pkl')

anno=pd.read_pickle(pickles+'testset_objects.pkl')
negs='/kaggle/input/arthropod-taxonomy-orders-object-detection-testset/ArTaxOr_TestSet/negatives/*.jpg'

nlist=glob.glob(negs, recursive=False)

fig = plt.figure(figsize=(16,24))

for i in range(len(nlist)//2):

    for j in range(2):

        axes = fig.add_subplot(len(nlist)//2, 2, 1+i*2+j)

        img, by = resize_image(nlist[i*2+j], 512, 512, stretch=False)

        scores, classes, boxes,_ = predict(img)

        plot_img_pred(img, axes, scores.squeeze(), boxes.squeeze(), classes.squeeze(), 'Prediction', by)
def pred_batch(idx, stretch):

    fig = plt.figure(figsize=(16,24))

    rows = 3

    for i in range(rows):

        img, by = resize_image(df.path.iloc[i+idx].replace('F:/', 'F:/'), 512, 512, stretch)

        axes = fig.add_subplot(rows, 2, 1+i*2)

        boxes = anno[anno.id == df.id.iloc[i+idx]][['label','xres', 'yres', 'xcenter', 'ycenter', 'width', 'height']]

        plot_img_gt(img, axes, boxes, stretch, 'Ground truth', by)

        img, by = resize_image(df.path.iloc[i+idx].replace('F:/', 'F:/'), 512, 512, stretch)

        scores, classes, boxes,_ = predict(img)

        axes = fig.add_subplot(rows, 2, 2+i*2)

        plot_img_pred(img, axes, scores.squeeze(), boxes.squeeze(), classes.squeeze(), 'Prediction', '') 
pred_batch(0, False)
pred_batch(3, False)
pred_batch(6, False)
pred_batch(12, False)