import numpy as np

import pandas as pd

import os

import json

import numpy as np

import cv2

from matplotlib import image as mpimg

from matplotlib import pyplot as plt

import pytesseract

from tqdm import tqdm_notebook

from pylab import rcParams

rcParams['figure.figsize'] = 16, 12

%config InlineBackend.figure_format = 'svg'
# считываем файл с разметкой и загружаем его как словарь

with open('/kaggle/input/chelhack/train.json', 'r') as f:

    annotation = json.load(f)
# так выглядит словарь с разметкой

# annotation
# пример, как можно итерироваться по словарю с разметкой

counter = 0

image_folder = '/kaggle/input/chelhack/train/'



for key, value in annotation.items():

    image_name = value['filename']

    print('id изображения', image_name)

    image_path = image_folder + image_name

    image = mpimg.imread(image_path)

    counter += 1

    for region in value['regions']:

        all_points_x = region['shape_attributes']['all_points_x']

        all_points_y = region['shape_attributes']['all_points_y']

        number = region['region_attributes']['description']

        print('x точки полигона', all_points_x)

        print('у точки полигона:', all_points_y)

        print('символы номера', number)

    print()

    if counter >= 3:

        break
# функция для показа изображения и разметки

def show(image_path, annotation, show_mask=True, show_number=True):

    rcParams['figure.figsize'] = 16, 12

    image = mpimg.imread(image_path)

    image_name = image_path.split('/')[-1]

    size = str(os.path.getsize(image_path))

    for i in range(len(annotation[image_name+size]['regions'])):

        number = annotation[image_name+size]['regions'][i]['region_attributes']['description']

        print(number)

        all_x = annotation[image_name+size]['regions'][i]['shape_attributes']['all_points_x']

        all_y = annotation[image_name+size]['regions'][i]['shape_attributes']['all_points_y']

        polygon = []

        for i in range(len(all_x)):

            polygon.append([all_x[i], all_y[i]])

        pts = np.array(polygon, np.int32)

        pts = pts.reshape((-1,1,2))

        cv2.polylines(image, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)

    plt.imshow(image)
image_path = '/kaggle/input/chelhack/train/61f67589ff4715a3c8fc5248fc999b31201906251002098100-full.jpg'

show(image_path, annotation)
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

img = mpimg.imread("/kaggle/input/chelhack/train/61f67589ff4715a3c8fc5248fc999b31201906251002098100-full.jpg")

plt.imshow(img);
plates = plate_cascade.detectMultiScale(img)



for (x,y,w,h) in plates:

    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) # рисуем бокс с детекцией

    cropped = img[y:y+h, x:x+w] # область с номером

plt.imshow(img);
# пример функции для локализации номера

def detect(image):

    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

    plates = plate_cascade.detectMultiScale(image)

    return plates
text = pytesseract.image_to_string(cropped) 

print(text)
# пример функции для распознавания символов

def recognize(cropped_image):

    number = pytesseract.image_to_string(cropped_image)

    return number
sample_submission = pd.read_csv('/kaggle/input/chelhack/sample_submission_val.csv')
PATH = '/kaggle/input/chelhack/val'

images_list = os.listdir(PATH)



for image_name in tqdm_notebook(images_list):

    image_path = os.path.join(PATH, image_name)

    image = mpimg.imread(image_path)

    plates = detect(image)

    for plate in plates:

        x, y, w, h = plate

        cropped_image = image[y:y+h, x:x+w]

        number = recognize(cropped_image)

        sample_submission.loc[sample_submission.ImageId == image_name, 'PredictionString'] = number
sample_submission.to_csv('test_submission.csv', index=False)