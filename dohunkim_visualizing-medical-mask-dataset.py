!pip install xmltodict

import os

import cv2

import matplotlib.pyplot as plt

import xmltodict

import random
def get_path(image_name):

    home_path = '/kaggle/input/medical-masks-dataset/'

    image_path = home_path + 'images/' + image_name

    

    if image_name[-4:] == 'jpeg':

        label_name = image_name[:-5] + '.xml'

    else:

        label_name = image_name[:-4] + '.xml'

    

    label_path = home_path + 'labels/' + label_name

        

    return  image_path, label_path





def parse_xml(label_path):

    x = xmltodict.parse(open(label_path , 'rb'))

    item_list = x['annotation']['object']

    

    # when image has only one bounding box

    if not isinstance(item_list, list):

        item_list = [item_list]

        

    result = []

    

    for item in item_list:

        name = item['name']

        bndbox = [(int(item['bndbox']['xmin']), int(item['bndbox']['ymin'])),

                  (int(item['bndbox']['xmax']), int(item['bndbox']['ymax']))]       

        result.append((name, bndbox))

    

    size = [int(x['annotation']['size']['width']), 

            int(x['annotation']['size']['height'])]

    

    return result, size





def visualize_image(image_name, bndbox=True):

    image_path, label_path = get_path(image_name)

    

    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    

    if bndbox:

        labels, size = parse_xml(label_path)

        thickness = int(sum(size)/400.)

        

        for label in labels:

            name, bndbox = label

            

            if name == 'good':

                cv2.rectangle(image, bndbox[0], bndbox[1], (0, 255, 0), thickness)

            elif name == 'bad':

                cv2.rectangle(image, bndbox[0], bndbox[1], (255, 0, 0), thickness)

            else: # name == 'none'

                cv2.rectangle(image, bndbox[0], bndbox[1], (0, 0, 255), thickness)

    

    plt.figure(figsize=(20, 20))

    plt.subplot(1, 2, 1)

    plt.axis('off')

    plt.title(image_name)

    plt.imshow(image)

    plt.show()
# name_list = os.listdir('/kaggle/input/medical-masks-dataset/images')

# names = random.sample(name_list, 5)



names = ['20200128150215888112.jpeg', '0602623232127-web-tete.jpg', '0_8w7mkX-PHcfMM5s6.jpeg']



for name in names:

    visualize_image(name)
visualize_image('shutterstock_289132226.jpg')

visualize_image('funnyMask_thumb_20200201_J_1024.jpg')
visualize_image('maxresdefault.jpg')
visualize_image('febfe125a0904416a97b9ff3be08517c.jpeg')