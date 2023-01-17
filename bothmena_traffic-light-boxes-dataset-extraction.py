import numpy as np

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

import cv2

import random

import os

import numpy as np

import pandas as pd

import zipfile

import shutil
!echo "daySequence1 frames = $(ls ../input/daySequence1/daySequence1/frames -l | wc -l)"

!echo "daySequence2 frames = $(ls ../input/daySequence2/daySequence2/frames -l | wc -l)"

!echo "nightSequence1 frames = $(ls ../input/nightSequence1/nightSequence1/frames -l | wc -l)"

!echo "nightSequence2 frames = $(ls ../input/nightSequence2/nightSequence2/frames -l | wc -l)"



!echo "day train ========"

!echo "dayTrain:dayClip1 frames = $(ls ../input/dayTrain/dayTrain/dayClip1/frames -l | wc -l)"

!echo "dayTrain:dayClip2 frames = $(ls ../input/dayTrain/dayTrain/dayClip2/frames -l | wc -l)"

!echo "dayTrain:dayClip3 frames = $(ls ../input/dayTrain/dayTrain/dayClip3/frames -l | wc -l)"

!echo "dayTrain:dayClip4 frames = $(ls ../input/dayTrain/dayTrain/dayClip4/frames -l | wc -l)"

!echo "dayTrain:dayClip5 frames = $(ls ../input/dayTrain/dayTrain/dayClip5/frames -l | wc -l)"

!echo "dayTrain:dayClip6 frames = $(ls ../input/dayTrain/dayTrain/dayClip6/frames -l | wc -l)"

!echo "dayTrain:dayClip7 frames = $(ls ../input/dayTrain/dayTrain/dayClip7/frames -l | wc -l)"

!echo "dayTrain:dayClip8 frames = $(ls ../input/dayTrain/dayTrain/dayClip8/frames -l | wc -l)"

!echo "dayTrain:dayClip9 frames = $(ls ../input/dayTrain/dayTrain/dayClip9/frames -l | wc -l)"

!echo "dayTrain:dayClip10 frames = $(ls ../input/dayTrain/dayTrain/dayClip10/frames -l | wc -l)"

!echo "dayTrain:dayClip11 frames = $(ls ../input/dayTrain/dayTrain/dayClip11/frames -l | wc -l)"

!echo "dayTrain:dayClip12 frames = $(ls ../input/dayTrain/dayTrain/dayClip12/frames -l | wc -l)"

!echo "dayTrain:dayClip13 frames = $(ls ../input/dayTrain/dayTrain/dayClip13/frames -l | wc -l)"



!echo "night train ========"

!echo "nightTrain:nightClip1 frames = $(ls ../input/nightTrain/nightTrain/nightClip1/frames -l | wc -l)"

!echo "nightTrain:nightClip2 frames = $(ls ../input/nightTrain/nightTrain/nightClip2/frames -l | wc -l)"

!echo "nightTrain:nightClip3 frames = $(ls ../input/nightTrain/nightTrain/nightClip3/frames -l | wc -l)"

!echo "nightTrain:nightClip4 frames = $(ls ../input/nightTrain/nightTrain/nightClip4/frames -l | wc -l)"

!echo "nightTrain:nightClip5 frames = $(ls ../input/nightTrain/nightTrain/nightClip5/frames -l | wc -l)"
# !ls ../input/Annotations/Annotations/daySequence1/  # frameAnnotationsBOX.csv  frameAnnotationsBULB.csv

# !ls ../input/Annotations/Annotations/daySequence2/  # frameAnnotationsBOX.csv  frameAnnotationsBULB.csv

# !ls ../input/Annotations/Annotations/dayTrain/dayClip[1 - 13]  # frameAnnotationsBOX.csv  frameAnnotationsBULB.csv

# !ls ../input/Annotations/Annotations/nightTrain/nightClip[1 - 5]  # frameAnnotationsBOX.csv  frameAnnotationsBULB.csv

# !ls ../input/Annotations/Annotations/nightSequence1/  # frameAnnotationsBOX.csv  frameAnnotationsBULB.csv

# !ls ../input/Annotations/Annotations/nightSequence2/ # frameAnnotationsBOX.csv  frameAnnotationsBULB.csv

box_ann_df = pd.read_csv('../input/Annotations/Annotations/dayTrain/dayClip10/frameAnnotationsBOX.csv', sep=';')

bulb_ann_df = pd.read_csv('../input/Annotations/Annotations/dayTrain/dayClip10/frameAnnotationsBULB.csv', sep=';')
# box_ann_df.groupby('Origin frame number').count()

box_ann_df.head()
bulb_ann_df.head()
dirs = [('daySequence1', None), ('daySequence2', None), ('nightSequence1', None), ('nightSequence2', None)]

for i in range(13):

    dirs.append(('dayTrain', 'dayClip{}'.format(i+1)))

for i in range(5):

    dirs.append(('nightTrain', 'nightClip{}'.format(i+1)))

def get_path(directory: str, frame_nbr: int, extension: str, sub_dir: str = None):

    if sub_dir is None:

        return '../input/{}/{}/frames/{}--{:05d}.{}'.format(directory, directory, directory, frame_nbr, extension)

    return '../input/{}/{}/{}/frames/{}--{:05d}.{}'.format(directory, directory, sub_dir, sub_dir, frame_nbr, extension)

    



def get_annotation_path(directory: str, sub_dir: str):

    if sub_dir is None:

        return '../input/Annotations/Annotations/{}/frameAnnotationsBOX.csv'.format(directory)

    return '../input/Annotations/Annotations/{}/{}/frameAnnotationsBOX.csv'.format(directory, sub_dir)

ext_fps = 15  # extraction fps (after how many frames extract the bounding box of an image)

bbox_margin = 5  # bounding box margin

output_dir = 'dataset'

edge_threshold = 10  # egde of bounding box threshold in pixels.
def square_bbox(x, y, width, height):

    """

    make a bbox square. the function assume that the square bbox can fit in the image.

    the function can deal with the case where it should pad the bbox with 5 pixels from left & right but

    there is only 3 pixels on the right of bbox.

    """

    if width < height:

        x -= min((height - width) // 2, x)

    if height < width:

        y -= min((width - height) // 2, y)

    width = max(width, height)

    height = width

    

    return x, y, width, height

def bbox_padding(x, y, edge, min_edge: int = None, expansion: int = None) -> tuple:

    """

    the function assumes that the bbox is a square.

    this perfoems a padding in two cases:

    - min_edge > 0: every bbox with edge < min_edge will add padding to it so its edge will become = min_edge

    - expansion > 1: will add padding to a bbox so its edge will be = edge * expansion

    

    Always either min_edge or expansion should be none and the other is valid (min_edge > 0 | expansion > 1)

    """

    if (min_edge is None and expansion is None) or (min_edge is not None and expansion is not None):

        raise ValueError('Only one of {min_edge, expansion} should be None')

    

    if min_edge is None:

        if expansion < 1:

            raise ValueError('expansion should be > 1')

        total_pad = round(edge * (expansion - 1))

        x -= min(total_pad // 2, x)

        y -= min(total_pad // 2, y)

        edge += total_pad

    else:

        if min_edge <= edge_threshold:

            raise ValueError('min_edge should be greater than edge_threshold = ' + str(edge_threshold))

        

        if edge < min_edge:

            total_pad = (min_edge - edge)

            x -= min(total_pad // 2, x)

            y -= min(total_pad // 2, y)

            edge = min_edge

    

    return x, y, edge, edge
img_dir = os.path.join(output_dir, 'images')

os.makedirs(img_dir, exist_ok=True)
# frame_filename: frame image filename

# bbox_filename: bbox image filename

# annotation_tag: class of the image

# sequence: video sequence: daySequence1, dayTrain, ...

# sub_sequence: none, dayClip10, nightClip4, ...

# x, y, width, height: bbox top-left point coordinates + width + height in the original image

# frame_nbr: frame_nbr in the sequence.

output_file = open(os.path.join(output_dir, 'annotations.csv'), 'w')

_ = output_file.write('frame_filename,bbox_filename,annotation_tag,sequence,sub_sequence,original_img,x,y,width,height,frame_nbr\n')
for directory, sub_dir in dirs:

    print('processing: {}{}'.format(directory, '' if sub_dir is None else '/' + sub_dir))

    

    ann_path = get_annotation_path(directory, sub_dir)

    ann_df = pd.read_csv(ann_path, sep=';')

    ann_df = ann_df.set_index(['Origin frame number', ann_df.index]).rename_axis(['frame', 'index'])

    unique_frames = ann_df.index.get_level_values(0).unique()

    nbr_frames = len(unique_frames)

    

    for frame_idx in range(0, nbr_frames, ext_fps):

        frame_nbr = unique_frames[frame_idx]

        frame_df = ann_df.loc[(frame_nbr, slice(None)), :]

        first_row = frame_df.iloc[0, :]

        extension = first_row['Filename'].split('.')[-1]

        img_path = get_path(directory, frame_nbr, extension, sub_dir)



        image = cv2.imread(img_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



        for bbox_idx, row in frame_df.iterrows():

            x, x_max = int(row['Upper left corner X']), int(row['Lower right corner X']) + 1

            y, y_max = int(row['Upper left corner Y']), int(row['Lower right corner Y']) + 1

            height = y_max - y

            width = x_max - x



            if min(height, width) < edge_threshold:

                continue



            x, y, height, width = square_bbox(x, y, height, width)

            x, y, height, width = bbox_padding(x, y, height, min_edge=20)

            x, y, height, width = bbox_padding(x, y, height, expansion=2)



            annotation = row['Annotation tag']

            bbox = image[y:y+height, x:x+width, :]



            bbox_fn = '{}_{}_{:05d}_{:05d}.{}'.format(directory, 'none' if sub_dir is None else sub_dir, *bbox_idx, extension)

            plt.imsave(os.path.join(img_dir, bbox_fn), bbox)



            frame_filename = os.path.basename(img_path)

            output_file.write('{},{},{},{},{},{},{},{},{},{}\n'.format(frame_filename, bbox_fn, annotation, directory, sub_dir, x, y, width, height, frame_nbr))



output_file.close()

print('finished processing')

!ls dataset/images -l | wc -l
def zipdir(path, ziph):

    # ziph is zipfile handle

    for root, dirs, files in os.walk(path):

        for file in files:

            ziph.write(os.path.join(root, file))

zipf = zipfile.ZipFile('output.zip', 'w', zipfile.ZIP_DEFLATED)

zipdir(output_dir, zipf)

zipf.close()
shutil.rmtree(output_dir)