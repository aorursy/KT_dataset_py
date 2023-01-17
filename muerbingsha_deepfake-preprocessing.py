import numpy as np

import pandas as pd

import cv2

from PIL import Image, ImageStat, ImageEnhance 

from math import sqrt

import matplotlib.pyplot as plt

from torchvision import transforms
BR_THRESHOLD = 95
transform = transforms.Compose([

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406],

                             std=[0.229, 0.224, 0.225])])
# get brightness



def get_brightness1(img_path):

    img = Image.open(img_path).convert('L')

    stat = ImageStat.Stat(img)

    return stat.mean[0]



def get_brightness2(img_path, method=1):

    img = Image.open(img_path).convert('RGB')

    

    all_b_1 = []

    all_b_2 = []

    all_b_3 = []



    for x in range(224):

        for y in range(224):

            R, G, B = img.getpixel((x, y))



            ##0 is dark (black) and 255 is bright (white)

            brightness = sum([R,G,B])/3 



            #Standard

            lum_a = (0.2126*R) + (0.7152*G) + (0.0722*B)



            #Percieved A

            lum_b = (0.299*R + 0.587*G + 0.114*B)



            all_b_1.append(brightness)

            all_b_2.append(lum_a)

            all_b_3.append(lum_b)

            

    if method == 1:

        return np.array(all_b_1).mean()

    elif method == 2:

        return np.array(all_b_2).mean()

    elif method == 3:

        return np.array(all_b_3).mean()

    
# dark

img_path = '../input/deepfake486326facescleaned/outputs2/train_sample_videos/aklqzsddfl-0-299.jpg'

img = cv2.imread(img_path)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
# normalize

img = transform(Image.fromarray(img))

plt.imshow(img.numpy().transpose(1, 2,0))
br1 = get_brightness1(img_path)

br2 = get_brightness2(img_path)

print(br1)

print(br2)
img = Image.open(img_path)

enhancer = ImageEnhance.Brightness(img)

enhanced_im = enhancer.enhance(BR_THRESHOLD/br1)

enhanced_im
img = transform(enhanced_im)

plt.imshow(img.numpy().transpose(1, 2,0))
# normal

img_path = '../input/deepfake486326facescleaned/outputs2/train_sample_videos/aknbdpmgua-0-149.jpg'

img = cv2.imread(img_path)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
print(get_brightness1(img_path))

print(get_brightness2(img_path))
# bright

img_path = '../input/deepfake486326facescleaned/outputs2/dfdc_train_part_45/bljpdykszf-0-0.jpg'

img = cv2.imread(img_path)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
img = transform(img)

plt.imshow(img.numpy().transpose(1, 2,0))
br1 = get_brightness1(img_path)

br2 = get_brightness2(img_path)

print(br1)

print(br2)
img = Image.open(img_path)

enhancer = ImageEnhance.Brightness(img)

enhanced_im = enhancer.enhance(BR_THRESHOLD/br1)

enhanced_im
img = transform(enhanced_im)

plt.imshow(img.numpy().transpose(1, 2,0))
# bright

img_path = '../input/deepfake486326facescleaned/outputs2/dfdc_train_part_21/opobaqiejv-0-134.jpg'

img = cv2.imread(img_path)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
br1 = get_brightness1(img_path)

br2 = get_brightness2(img_path)

print(br1)

print(br2)
img = Image.open(img_path)

enhancer = ImageEnhance.Brightness(img)

enhanced_im = enhancer.enhance(1 if BR_THRESHOLD/br1 < 1 else BR_THRESHOLD/br1) # < 1

enhanced_im
img_path = '../input/deepfake486326facescleaned/outputs2/dfdc_train_part_21/opobaqiejv-0-134.jpg'

img = cv2.imread(img_path)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
img = Image.open(img_path).convert('L') # to grayscale

array = np.asarray(img, dtype=np.int32)



gy, gx = np.gradient(array)

gnorm = np.sqrt(gx**2 + gy**2)

sharpness = np.average(gnorm)

sharpness
img = Image.open(img_path)

enhancer = ImageEnhance.Sharpness(img)

enhanced_im = enhancer.enhance(10.0)

enhanced_im
img = transform(img)

plt.figure(1)

plt.imshow(img.numpy().transpose(1, 2,0))



img = transform(enhanced_im)

plt.figure(2)

plt.imshow(img.numpy().transpose(1, 2,0))
def load_ben_color(path, sigmaX=10 ):

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)

        

    return image
img = load_ben_color('../input/deepfake486326facescleaned/outputs2/dfdc_train_part_45/bljpdykszf-0-0.jpg')

plt.imshow(img)
img = transform(Image.fromarray(img))

plt.imshow(img.numpy().transpose(1, 2,0))
# another example

img = load_ben_color('../input/deepfake486326facescleaned/outputs2/train_sample_videos/aklqzsddfl-0-299.jpg')

img = transform(Image.fromarray(img))

plt.imshow(img.numpy().transpose(1, 2,0))