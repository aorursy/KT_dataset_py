import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2 # image processing



from matplotlib import pyplot as plt # data visualization



# making sure result is reproducible

SEED = 2019

np.random.seed(SEED)
def read_image(image):

    '''

        Simply read a single image and convert it RGB in opencv given its filename.

    '''



    return cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)





def apply_ben_preprocessing(image):

    '''

        Apply Ben's preprocessing on a single image in opencv format

    '''

    

    return cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), 10), -4, 128)





def apply_denoising(image):

    '''

        Apply denoising on a single image given it in opencv format.

        Denoising is done using twice the recommended strength from opencv docs.

    '''

    

    return cv2.fastNlMeansDenoisingColored(image, None, 20, 20, 7, 21)
train_df = pd.read_csv('../input/train.csv')

samples = train_df.sample(n=10)



for ID in samples['image_id']:

    filename = '../input/train_images/{}.jpg'.format(ID)

    

    img = read_image(filename)

    before = apply_ben_preprocessing(img)

    after = apply_denoising(before)



    fig, ax = plt.subplots(1, 2, figsize=(16, 20))

    ax[0].imshow(before)

    ax[1].imshow(after)