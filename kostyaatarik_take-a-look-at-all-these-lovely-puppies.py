from glob import glob

from IPython.core.display import display, HTML

import numpy as np

import xml.etree.ElementTree as ET

import cv2

import imageio





def get_bboxes(annotation_file):

    '''Extract and return bounding boxes from annotation file.'''

    bboxes = []

    objects = ET.parse(annotation_file).getroot().findall('object')

    for obj in objects:

        bbox = obj.find('bndbox')

        bboxes.append(tuple(int(bbox.find(_).text) for _ in ('xmin', 'ymin', 'xmax', 'ymax')))

    return bboxes





def center_crop(image):

    h, w = image.shape[:2]

    s = min(h, w)

    return image[(h-s)//2:(h-s)//2 + s,(w-s)//2:(w-s)//2 + s]





for breed in sorted(glob('../input/annotation/Annotation/*'), key=lambda breed: len(glob(breed + '/*')), reverse=True):

    breed_id, breed_name = breed.split('/')[-1].split('-', 1)

    breed_name = ' '.join(s.capitalize() for s in breed_name.replace('_', ' ').replace('-', ' ').split())

    annotations = glob(breed + '/*')

    np.random.shuffle(annotations)

    dogs = []

    for annotation_file in annotations[:11]:

        bboxes = get_bboxes(annotation_file)

        img_path = '../input/all-dogs/all-dogs/{}.jpg'.format(annotation_file.split('/')[-1])

        try:

            image = imageio.imread(img_path)

            for xmin, ymin, xmax, ymax in bboxes:

                dogs.append(cv2.resize(center_crop(image[ymin:ymax, xmin:xmax, :]), (128, 128)))

        except Exception as e:

            pass

    np.random.shuffle(dogs)

    dogs = np.hstack(dogs[:10])

    imageio.imwrite(f'{breed_name}.png', dogs)

    display(HTML('<div style="text-align:center"><span title="{1}"><a href="http://vision.stanford.edu/aditya86/ImageNetDogs/{0}.html"><b>{1}</b></a>, {2} samples</span></div>'.format(breed_id, breed_name, len(annotations))))

    display(HTML(f'<img src="{breed_name}.png" alt="Examples of {breed_name}" width="100%">'))