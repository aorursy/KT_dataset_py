#csv2xml

import numpy as np # linear algebra

import pandas as pd



path = '/kaggle/working/annotation'

os.makedirs(path) 

import csv

import os, ast

from xml.dom import minidom



# 文件路径

xml_dir = '/kaggle/working/annotation'

csv_filename = os.path.join('/kaggle/input/global-wheat-detection/', 'train.csv')

# 逐行读取csv文件

def create_xml(filename, bboxs):

    width = 1024

    height = 1024

    depth = 3

    # 1.创建DOM树对象

    dom = minidom.Document()

    # 2.创建根节点。每次都要用DOM对象来创建任何节点。

    root_node = dom.createElement('annotation')

    # 3.用DOM对象添加根节点

    dom.appendChild(root_node)



    filename_node = dom.createElement('filename')

    root_node.appendChild(filename_node)

    # 也用DOM创建文本节点，把文本节点（文字内容）看成子节点

    name_text = dom.createTextNode(filename)

    # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点

    filename_node.appendChild(name_text)



    # size

    size_node = dom.createElement('size')

    root_node.appendChild(size_node)

    width_node = dom.createElement('width')

    height_node = dom.createElement('height')

    depth_node = dom.createElement('depth')

    # width

    size_node.appendChild(width_node)

    width_text = dom.createTextNode(str(width))

    width_node.appendChild(width_text)

    # height

    size_node.appendChild(height_node)

    height_text = dom.createTextNode(str(height))

    height_node.appendChild(height_text)

    # depth

    size_node.appendChild(depth_node)

    depth_text = dom.createTextNode(str(depth))

    depth_node.appendChild(depth_text)



    for bbox in bboxs:

        # 创建obejct

        object_node = dom.createElement('object')

        root_node.appendChild(object_node)

        # 创建类别name

        name_node = dom.createElement('name')

        name_text = dom.createTextNode('Wheat')

        name_node.appendChild(name_text)

        object_node.appendChild(name_node)

        # 创建bndbox

        # bbox [xmin, ymin, width, height]

        bbox = ast.literal_eval(bbox)

        xmin, ymin = bbox[0], bbox[1]

        xmax, ymax = xmin + bbox[2], ymin + bbox[3]



        bndbox = dom.createElement('bndbox')

        object_node.appendChild(bndbox)

        # xmin

        xmin_node = dom.createElement('xmin')

        xmin_text = dom.createTextNode(str(xmin))

        xmin_node.appendChild(xmin_text)

        bndbox.appendChild(xmin_node)

        # ymin

        ymin_node = dom.createElement('ymin')

        ymin_text = dom.createTextNode(str(ymin))

        ymin_node.appendChild(ymin_text)

        bndbox.appendChild(ymin_node)

        # xmax

        xmax_node = dom.createElement('xmax')

        xmax_text = dom.createTextNode(str(xmax))

        xmax_node.appendChild(xmax_text)

        bndbox.appendChild(xmax_node)

        # ymax

        ymax_node = dom.createElement('ymax')

        ymax_text = dom.createTextNode(str(ymax))

        ymax_node.appendChild(ymax_text)

        bndbox.appendChild(ymax_node)



    # 每一个结点对象（包括dom对象本身）都有输出XML内容的方法，如：toxml()--字符串, toprettyxml()--美化树形格式。

    try:

        with open(os.path.join(xml_dir, filename) + '.xml', 'w', encoding='UTF-8') as fh:

            # 4.writexml()第一个参数是目标文件对象，第二个参数是根节点的缩进格式，第三个参数是其他子节点的缩进格式，

            # 第四个参数制定了换行格式，第五个参数制定了xml内容的编码。

            dom.writexml(fh, indent='', addindent='\t', newl='\n', encoding='UTF-8')

#             print('写入xml OK!')

    except Exception as err:

        print('错误信息：{0}'.format(err))

def main():

    with open(csv_filename, 'r', encoding="utf-8") as csvfile:

        reader = csv.DictReader(csvfile)

        # 自动获取第一张照片的文件名，并设置为last_image

        i = 0

        for row in reader:

            if i == 1:

                last_image = row['image_id']

                break

            i += 1

        # print(last_image)

        img_num = 1

        bboxs = []

        for row in reader:

            if row['image_id'] == last_image:

                # 叠加bbox [xmin, ymin, width, height]

                bboxs.append(row['bbox'])

            elif row['image_id'] != last_image:

                bboxs.append(row['bbox'])

                # 创建xml文件

                create_xml(last_image, bboxs)

                last_image = row['image_id']

                img_num += 1

                # 重置bbox

                bboxs.clear()



        print(img_num)

main()
# 将xml文件数据进行读取

import os, glob



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

import tensorflow as tf

from tensorflow import keras



tf.random.set_seed(2234)

np.random.seed(2234)



# print(tf.__version__)

# print(tf.test.is_gpu_available())



obj_names = ('Wheat')

img_dir = '/kaggle/input/global-wheat-detection/train'

ann_dir = '/kaggle/working/annotation'



import xml.etree.ElementTree as ET



# 1.1

def parse_annotation(img_dir, ann_dir, labels):

    # img_dir: image path

    # ann_dir: annotation xml file path

    # labels: ('wheat')

    # 从 annotation info from xml file

    imgs_info = []

    max_boxes = 0

    for ann in os.listdir(ann_dir):

        tree = ET.parse(os.path.join(ann_dir, ann))

        img_info = dict()

        img_info['object'] = []

        boxes_counter = 0

        for elem in tree.iter():



            if 'filename' in elem.tag:

                img_info['filename'] = os.path.join(img_dir, elem.text)

            if 'width' in elem.tag:

                img_info['width'] = int(elem.text)

                assert img_info['width'] == 1024

            if 'height' in elem.tag:

                img_info['height'] = int(elem.text)

                assert img_info['height'] == 1024



            if 'object' in elem.tag or 'part' in elem.tag:

                # x1-y1-x2-y2-label 左上角，右下脚坐标

                object_info = [0., 0., 0., 0., 0.]

                boxes_counter += 1

                for attr in list(elem):

                    if 'name' in attr.tag:

                        label = labels.index(attr.text) + 1

                        object_info[4] = label

                    if 'bndbox' in attr.tag:

                        for pos in list(attr):

                            if 'xmin' in pos.tag:

                                object_info[0] = float(pos.text)

                            if 'ymin' in pos.tag:

                                object_info[1] = float(pos.text)

                            if 'xmax' in pos.tag:

                                object_info[2] = float(pos.text)

                            if 'ymax' in pos.tag:

                                object_info[3] = float(pos.text)

                img_info['object'].append(object_info)

        imgs_info.append(img_info)  # filename,w/h/box_info

        # (N,5) = (max_objects_num,5)

        if boxes_counter > max_boxes:

            max_boxes = boxes_counter

        # the maximum boxes number is max_boxes

    # [3372,116,5]

    boxes = np.zeros([len(imgs_info), max_boxes, 5])

    print(boxes.shape)

    imgs = []  # filename list

    for i, img_info in enumerate(imgs_info):

        # [N,5]

        img_boxes = np.array(img_info['object'])

        # overwrite the N boxes info

        boxes[i, :img_boxes.shape[0]] = img_boxes

        imgs.append(img_info['filename']+'.jpg')

    #         print(img_info['filename'], boxes[i:1])

    # imgs :list of image path [b]

    # boxes:[b,116,5]

    return imgs, boxes





def preprocess(img, img_boxes):

    # img:string

    # img_boxes:[116,5]

    x = tf.io.read_file(img)

    x = tf.image.decode_png(x, channels=3)

    x = tf.image.convert_image_dtype(x, tf.float32)

    return x, img_boxes





# 1.2

def get_dataset(img_dir, ann_dir, batchsz):

    # return tf dataset

    # [b],boxes[b,116,5]

    imgs, boxes = parse_annotation(img_dir, ann_dir, obj_names)

    db = tf.data.Dataset.from_tensor_slices((imgs, boxes))

    db = db.shuffle(100).map(preprocess).batch(batchsz).repeat()

    print('db_images', len(imgs))

    return db





train_db = get_dataset(img_dir, ann_dir, batchsz=64)

print(train_db)



# 1.3 visual the db

from matplotlib import pyplot as plt

from matplotlib import patches



def db_visualize(db):

    # [b,1024,1024,3]

    # imgs_boxes [b,116,5]

    imgs, imgs_boxes = next(iter(db))

    #在这里设置你想查看的图像

    img, img_boxes = imgs[10], imgs_boxes[10]

    f, ax1 = plt.subplots(1)

    # display the image,[1024,2014,3]

    ax1.imshow(img)

    for x1, y1, x2, y2, l in img_boxes:

        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

        w = x2 - x1

        h = y2 - y1

        if l == 1:

            color = (0, 1, 0)

        else:

            break

        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor='none')

        ax1.add_patch(rect)





db_visualize(train_db)
