import xml.etree.ElementTree as ET

import pandas as pd

import numpy as np
# get image ids

marking = pd.read_csv('../input/global-wheat-detection/train.csv')



bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))

for i, column in enumerate(['x', 'y', 'w', 'h']):

    marking[column] = bboxs[:,i]

marking.drop(columns=['bbox'], inplace=True)





df_imgs = marking['image_id']

image_ids = df_imgs.unique()

# for a check

print(image_ids)

print(len(image_ids))
width, height, depth = 1024, 1024, 3


def trans_to_XML():

    for imindex in range(len(image_ids)): # for each image



        img_mark = marking[marking['image_id'] == image_ids[imindex]]

        annotation = ET.Element('annotation')

        

#         print (image_ids[imindex])

#         print(img_mark)

        for  row in img_mark.itertuples():

#             print(row)

            ET.SubElement(annotation, 'folder').text = 'train'

            ET.SubElement(annotation, 'filename').text = image_ids[imindex] + '.jpg'

            size = ET.SubElement(annotation, 'size')

            ET.SubElement(size, 'width').text = str(width)

            ET.SubElement(size, 'height').text = str(height)

            ET.SubElement(size, 'depth').text = str(depth)

            obj = ET.SubElement(annotation, 'object')

            ET.SubElement(obj, 'name').text = 'wheat head'

            ET.SubElement(obj, 'pose').text = 'Unspecified'

            ET.SubElement(obj, 'truncated').text = '0'

            ET.SubElement(obj, 'difficult').text = '0'

            bbox = ET.SubElement(obj, 'bndbox')

#             print(bbox.text)

            ET.SubElement(bbox, 'xmin').text = str(row[5])

            ET.SubElement(bbox, 'ymin').text = str(row[6])

            ET.SubElement(bbox, 'xmax').text = str(row[5] + row[7])

            ET.SubElement(bbox, 'ymax').text = str(row[6] + row[8])



        tree = ET.ElementTree(annotation)

        # here is the path you want to save, I used my train picture fold

        outputfold = ''

        tree.write(outputfold + image_ids[imindex] + '.xml', encoding='utf-8')

        if imindex % 100 == 0:

            print(imindex, image_ids[imindex] + ' finished', end='\t')

            if imindex % 300 == 0:

                print()
trans_to_XML()
    