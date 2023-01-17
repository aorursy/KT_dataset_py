import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
print(os.listdir("../input"))
%matplotlib inline
# Any results you write to the current directory are saved as output.
class BlurAwareCrop():
    def __init__(self, prob=0.7, blur_thres=200, min_crop=70, return_size=101):
        self.prob = prob
        self.blur_thres = blur_thres
        self.min_crop = min_crop
        self.return_size = return_size
        self.tr = None
    
    # reference: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    def sharp_measure(self, img_pil):
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return cv2.Laplacian(img_cv, cv2.CV_64F).var()
    
    def __call__(self, img):
        '''
        if given image has RGB mode(salt image), compute the sharpness of image using cv and setup transforms to be applied
        otherwise, if mask is given, just applies same transform again.
        '''
        if img.mode == 'RGB':
            if self.sharp_measure(img) > self.blur_thres and np.random.rand() < self.prob:
                crop_size = np.random.randint(self.min_crop, self.return_size)
                self.tr = transforms.Compose([
                    transforms.RandomCrop(crop_size),
                    transforms.Resize(self.return_size)
                ])
            else:
                self.tr = transforms.Compose([])
        return self.tr(img)
tr = BlurAwareCrop()
fnames = pd.read_csv('../input/train.csv', usecols=['id'])
def show_example(index):
    img = Image.open(f'../input/train/images/{fnames.id[index]}.png')
    mask = Image.open(f'../input/train/masks/{fnames.id[index]}.png')
    sharpness = tr.sharp_measure(img)
    print(f"image sharpness: {sharpness}")
    if sharpness > tr.blur_thres:
        print(f"image is sharp enough, cropping is applied with probability {tr.prob}")
    else:
        print("image is blurry, cropping will not applied")
    
    plt.figure(figsize=(16, 9))
    
    plt.subplot(141)
    plt.title('image before transform')
    plt.imshow(img)
    
    plt.subplot(142)
    plt.title('mask before transform')
    plt.imshow(mask)

    plt.subplot(143)
    plt.title('image after transform')
    plt.imshow(tr(img))
    
    plt.subplot(144)
    plt.title('mask after transform')
    plt.imshow(tr(mask))
    plt.show()
show_example(15)
show_example(19)
