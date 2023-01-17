# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

    #    print(os.path.join(dirname, filename))

    print(dirname)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

%matplotlib inline

import cv2
train_csv = '../input/global-wheat-detection/train.csv'

data = pd.read_csv(train_csv)

data
data.isnull().any()
data.info()
data.describe()
print(f'Total number of train images: {data.image_id.nunique()}')
# Extract bbox column to xmin, ymin, width, height, then create xmax, ymax, and area columns



data[['xmin','ymin','w','h']] = pd.DataFrame(data.bbox.str.strip('[]').str.split(',').tolist()).astype(float)

data['xmax'], data['ymax'], data['area'] = data['xmin'] + data['w'], data['ymin'] + data['h'], data['w'] * data['h']

data.drop(['bbox'], axis=1, inplace= True)

data
DATA_DIR = '../input/global-wheat-detection/train/'
def show_image(image_id):

    

    fig, ax = plt.subplots(1, 2, figsize = (24, 24))

    ax = ax.flatten()

    

    bbox = data[data['image_id'] == image_id ]

    img_path = os.path.join(DATA_DIR, image_id + '.jpg')

    

    image = cv2.imread(img_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    image /= 255.0

    image2 = image

    

    ax[0].set_title('Original Image')

    ax[0].imshow(image)

    

    for idx, row in bbox.iterrows():

        x1 = row['xmin']

        y1 = row['ymin']

        x2 = row['xmax']

        y2 = row['ymax']

        label = row['source']

        

        cv2.rectangle(image2, (int(x1),int(y1)), (int(x2),int(y2)), (255,255,255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(image2, label, (int(x1),int(y1-10)), font, 0.8, (255,255,255), 2)

    

    ax[1].set_title('Image with Bondary Box')

    ax[1].imshow(image2)



    plt.show()
show_image(data.image_id.unique()[90])
show_image(data.image_id.unique()[300])
show_image(data.image_id.unique()[1900])
show_image(data.image_id.unique()[2500])
def augument_bbox(width, height, xmin, ymin, xmax, ymax, aug_type=90):

    bbh = xmax - xmin

    bbw = ymax - ymin

    if aug_type == 90:

        ymin = xmin

        xmin = height - ymax

        xmax = xmin + bbw

        ymax = ymin + bbh

    if aug_type == 180:

        xmin = width - xmax

        ymin = height - ymax

        xmax = xmin + bbh

        ymax = ymin + bbw

    if aug_type == 270:

        xmin = ymin

        ymin = width - xmax

        xmax = xmin + bbw

        ymax = ymin + bbh

    if aug_type == 'Horizontal':

        ymin = ymin

        xmin = width - xmax

        xmax = xmin + bbh

        ymax = ymin + bbw

    if aug_type == 'Vertical':

        xmin = xmin

        ymin = height - ymax

        xmax = xmin + bbh

        ymax = ymin + bbw

    return xmin, ymin, xmax, ymax
def show_aug_images(image_id, aug_types):

    for aug_type in aug_types:



        fig, ax = plt.subplots(1, 2, figsize = (24, 24))

        ax = ax.flatten()



        bbox = data[data['image_id'] == image_id ]

        img_path = os.path.join(DATA_DIR, image_id + '.jpg')



        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0

        

        if aug_type == 90:

            image2 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            color = (255, 0, 0)

        if aug_type == 180:

            image2 = cv2.rotate(image, cv2.ROTATE_180)

            color = (0, 255, 0)

        if aug_type == 270:

            image2 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            color = (0, 0, 255)

        if aug_type == 'Horizontal':

            image2 = cv2.flip(image, 1)

            color = (255, 255, 0)

        if aug_type == 'Vertical':

            image2 = cv2.flip(image, 0)

            color = (0, 255, 255)

        

        for idx, row in bbox.iterrows():

            x1 = row['xmin']

            y1 = row['ymin']

            x2 = row['xmax']

            y2 = row['ymax']

            label = row['source']



            cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), (255,255,255), 4)

            if aug_type == 90:

                cv2.circle(image, (int(x1), int(y2)), 6, (0, 0, 255), -1)

            if aug_type == 180:

                cv2.circle(image, (int(x2), int(y2)), 6, (255, 0, 0), -1)

            if aug_type == 270:

                cv2.circle(image, (int(x2), int(y1)), 6, (0, 255, 0), -1)

            if aug_type == 'Horizontal':

                cv2.circle(image, (int(x2), int(y1)), 6, (0, 0, 255), -1)

            if aug_type == 'Vertical':

                cv2.circle(image, (int(x1), int(y2)), 6, (255, 0, 0), -1)



        ax[0].set_title('Original Image with Bondary Boxes')

        ax[0].imshow(image)



        for idx, row in bbox.iterrows():

            x1 = row['xmin']

            y1 = row['ymin']

            x2 = row['xmax']

            y2 = row['ymax']

            width = row['width'] 

            height = row['height']

            label = row['source']

            

            x1, y1, x2, y2 = augument_bbox(int(width), int(height), int(x1), int(y1), int(x2), int(y2), aug_type)

            

            cv2.rectangle(image2, (int(x1),int(y1)), (int(x2),int(y2)), color, 4)

            if aug_type == 90:

                cv2.circle(image2, (int(x1), int(y1)), 6, (0, 0, 255), -1)

            if aug_type == 180:

                cv2.circle(image2, (int(x1), int(y1)), 6, (255, 0, 0), -1)

            if aug_type == 270:

                cv2.circle(image2, (int(x1), int(y1)), 6, (0, 255, 0), -1)

            if aug_type == 'Horizontal':

                cv2.circle(image2, (int(x1), int(y1)), 6, (0, 0, 255), -1)

            if aug_type == 'Vertical':

                cv2.circle(image2, (int(x1), int(y1)), 6, (255, 0, 0), -1)



        ax[1].set_title(str(aug_type) + ' (degrees rotated/Flipped) Agumented Image with Bondary Boxes')

        ax[1].imshow(image2)



        plt.show()
aug_types = [90, 180, 270, 'Horizontal', 'Vertical']

show_aug_images(data.image_id.unique()[90], aug_types)
show_aug_images(data.image_id.unique()[300], aug_types)
show_aug_images(data.image_id.unique()[1900], aug_types)
show_aug_images(data.image_id.unique()[2500], aug_types)