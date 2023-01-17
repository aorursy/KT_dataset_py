!pip install tensorflow==2.0.0a0
!wget https://github.com/bonlime/keras-deeplab-v3-plus/archive/master.zip

!unzip master.zip



import sys

sys.path.insert(0, 'keras-deeplab-v3-plus-master')
import numpy as np

from PIL import Image

from matplotlib import pyplot as plt

%pylab inline



from model import Deeplabv3
deeplab_model = Deeplabv3()
def get_mask(image, model):

    trained_image_width=512 

    mean_subtraction_value=127.5



    # add 3-th dimension if needed

    if len(image.shape) == 2:

        image = np.tile(image[..., None], (1, 1, 3))

        

    # resize to max dimension of images from training dataset

    w, h, _ = image.shape

    ratio = float(trained_image_width) / np.max([w, h])

    resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))



    # apply normalization for trained dataset images

    resized_image = (resized_image / mean_subtraction_value) - 1.



    # pad array to square image to match training images

    pad_x = int(trained_image_width - resized_image.shape[0])

    pad_y = int(trained_image_width - resized_image.shape[1])

    resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')



    # make prediction

    res = model.predict(np.expand_dims(resized_image, 0))

    labels = np.argmax(res.squeeze(), -1)



    # remove padding and resize back to original image

    if pad_x > 0:

        labels = labels[:-pad_x]

    if pad_y > 0:

        labels = labels[:, :-pad_y]

    labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))

    

    return (labels == 15).astype('uint8')
image = np.array(Image.open('keras-deeplab-v3-plus-master/imgs/image3.jpg'))

mask = get_mask(image, deeplab_model)

plt.imshow(mask)
plt.figure(figsize=(15,10))

img = np.array(Image.open("../input/coco2017/train2017/train2017/000000281563.jpg"))

label = np.array(Image.open("../input/coco2017/stuffthingmaps_trainval2017/train2017/000000281563.png"))



plt.subplot(1,3, 1)

plt.imshow(img)

plt.title("Image")

plt.subplot(1,3, 2)

plt.imshow(label < 1)

plt.title("Label")

mask = get_mask(img, deeplab_model)

plt.subplot(1,3, 3)

plt.imshow(mask)

plt.title("Predict")
deeplab_model = Deeplabv3(backbone='xception', OS=8)
plt.figure(figsize=(15,10))

img = np.array(Image.open("../input/coco2017/train2017/train2017/000000281563.jpg"))

label = np.array(Image.open("../input/coco2017/stuffthingmaps_trainval2017/train2017/000000281563.png"))



plt.subplot(1,3, 1)

plt.imshow(img)

plt.title("Image")

plt.subplot(1,3, 2)

plt.imshow(label < 1)

plt.title("Label")

mask = get_mask(img, deeplab_model)

plt.subplot(1,3, 3)

plt.imshow(mask)

plt.title("Predict")
import pandas as pd

sample_submission = pd.read_csv('../input/sf-dl-2-person-segmentation/sample-submission.csv')
# кодирование маски в EncodedPixels

def mask_to_rle(mask):

    mask_flat = mask.flatten('F')

    flag = 0

    rle_list = list()

    for i in range(mask_flat.shape[0]):

        if flag == 0:

            if mask_flat[i] == 1:

                flag = 1

                starts = i+1

                rle_list.append(starts)

        else:

            if mask_flat[i] == 0:

                flag = 0

                ends = i

                rle_list.append(ends-starts+1)

    if flag == 1:

        ends = mask_flat.shape[0]

        rle_list.append(ends-starts+1)

    #sanity check

    if len(rle_list) % 2 != 0:

        print('NG')

    if len(rle_list) == 0:

        rle = np.nan

    else:

        rle = ' '.join(map(str,rle_list))

    return rle
submit_rle_arr = []



for img_id in sample_submission.ImageId.values:

    image = np.array(Image.open(f'../input/coco2017/val2017/{img_id}'))

    mask_out = get_mask(image, deeplab_model)

    rle = mask_to_rle(mask_out) 

    submit_rle_arr.append(rle)
sample_submission['EncodedPixels'] = submit_rle_arr

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head()