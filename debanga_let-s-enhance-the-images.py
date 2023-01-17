import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
from tqdm.notebook import tqdm
import random
import cv2
plt.rcParams['figure.figsize'] = [15,8]
DATA_DIR = '/kaggle/input/prostate-cancer-grade-assessment/'
data = pd.read_csv(DATA_DIR + 'train.csv')
data_karolinska = data[data.data_provider=="karolinska"].reset_index().drop(columns=['index'])
data_radboud = data[data.data_provider=="radboud"].reset_index().drop(columns=['index'])
def show_grid(dataframe):
    N = len(dataframe)
    for i in range(3):
        for j in range(3):
            plt.subplot(3,3,j+3*i+1)
            img = skimage.io.MultiImage(DATA_DIR + 'train_images/' + dataframe.image_id[int(N*random.random())] + '.tiff')
            plt.imshow(img[2])
            plt.axis('off')
print('Radboud Samples')
show_grid(data_radboud)
print('Karolinska Samples')
show_grid(data_karolinska)
def enhance_image(image, contrast=1, brightness=15):
    """
    Enhance constrast and brightness of images
    """
    img_enhanced = cv2.addWeighted(image, contrast, image, 0, brightness)
    return img_enhanced

factor = 4 
channel = 1# using 2nd channel
img = skimage.io.MultiImage(DATA_DIR + 'train_images/' + data_radboud.image_id[50] + '.tiff')[channel][factor*0:factor*300,factor*0:factor*300]
enhnaced_img = enhance_image(img)
img_concat = np.concatenate((img, enhnaced_img), axis=1)
plt.imshow(img_concat);plt.axis('off');
plt.title('A RADBOUD SAMPLE \n\n [Left] Before Image Enhancement, [Right] After Image Enhancement')
plt.show();

img = skimage.io.MultiImage(DATA_DIR + 'train_images/' + data_karolinska.image_id[50] + '.tiff')[channel][factor*0:factor*1300,factor*0:factor*1300]
enhnaced_img = enhance_image(img)
img_concat = np.concatenate((img, enhnaced_img), axis=1)
plt.imshow(img_concat);plt.axis('off');
plt.title('A KAROLINSKA SAMPLE \n\n [Left] Before Image Enhancement, [Right] After Image Enhancement')
plt.show();
def RGB_histogram_equalization(img):
    """ Histogram Equalization of 3-Channel images"""
    equalized_image1 = cv2.equalizeHist(img[:,:,0])
    equalized_image2 = cv2.equalizeHist(img[:,:,1])
    equalized_image3 = cv2.equalizeHist(img[:,:,2])
    return cv2.merge((equalized_image1,equalized_image2,equalized_image3))

factor = 4 
channel = 1# using 2nd channel
img = skimage.io.MultiImage(DATA_DIR + 'train_images/' + data_radboud.image_id[50] + '.tiff')[channel][factor*0:factor*300,factor*0:factor*300]
img = enhance_image(img)
equalized_image = RGB_histogram_equalization(img)
img_concat = np.concatenate((img, equalized_image),axis=1)
plt.imshow(img_concat);plt.axis('off');
plt.title('A RADBOUD SAMPLE \n\n [Left] Before Histogram Equalization, [Right] After Histogram Equalization')
plt.show();

img = skimage.io.MultiImage(DATA_DIR + 'train_images/' + data_karolinska.image_id[50] + '.tiff')[1][factor*0:factor*1300,factor*0:factor*1300]
img = enhance_image(img)
equalized_image = RGB_histogram_equalization(img)
img_concat = np.concatenate((img, equalized_image),axis=1)
plt.imshow(img_concat);plt.axis('off');
plt.title('A KAROLINSKA SAMPLE \n\n [Left] Before Histogram Equalization, [Right] After Histogram Equalization')
plt.show();
def unsharp_masking(img):
    """ Unsharp masking of an RGB image"""
    img_gaussian = cv2.GaussianBlur(img, (21,21), 10.0)
    return cv2.addWeighted(img, 1.8, img_gaussian, -0.8, 0, img)

factor = 4
img = skimage.io.MultiImage(DATA_DIR + 'train_images/' + data_radboud.image_id[500] + '.tiff')[1][factor*150:factor*250,factor*150:factor*250]
img = enhance_image(img)
unsharp_image = unsharp_masking(img.copy())
img_concat = np.concatenate((img, unsharp_image),axis=1)
plt.imshow(img_concat);plt.axis('off');
plt.title('A RADBOUD SAMPLE \n\n [Left] Before Unsharp Masking, [Right] After Unsharp Masking')
plt.show();

img = skimage.io.MultiImage(DATA_DIR + 'train_images/' + data_karolinska.image_id[500] + '.tiff')[1][factor*600:factor*1000,factor*600:factor*1000]
img = enhance_image(img)
unsharp_image = unsharp_masking(img.copy())
img_concat = np.concatenate((img, unsharp_image),axis=1)
plt.imshow(img_concat);plt.axis('off');
plt.title('A KAROLINSKA SAMPLE \n\n [Left] Before Unsharp Masking, [Right] Unsharp Masking')
plt.show();