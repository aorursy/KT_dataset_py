import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
print('imports complete')
PATH = "../input/shapes/"
IMG_SIZE = 64
Shapes = ["circle", "square", "triangle", "star"]
Labels = []
Dataset = []

all_shapes = []

# From kernel: https://www.kaggle.com/smeschke/load-data
for shape in Shapes:
    idx = 0
    temporary_list = []
    print("Getting data for: ", shape)
    #iterate through each file in the folder
    for path in os.listdir(PATH + shape)[:20]:
        #add the image to the list of images
        image = cv2.imread(PATH + shape + '/' + path,0)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        Dataset.append(image)
        temporary_list.append(image)
        #add an integer to the labels list 
        Labels.append(Shapes.index(shape))
    all_shapes.append(temporary_list)
    idx += 1

print("\nDataset Images size:", len(Dataset))
print("Image Shape:", Dataset[0].shape)
print("Labels size:", len(Labels))        
import random
length = len(all_shapes[0])
img_a = all_shapes[3][random.randint(0, length-1)]
for i in range(4):    
    img_b = all_shapes[i][random.randint(0, length-1)]
    m1 = cv2.matchShapes(img_a, img_b,cv2.CONTOURS_MATCH_I2,0)
    # Two subplots, unpack the axes array immediately
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.imshow(img_a)
    ax1.axis('off')
    ax2.imshow(img_b)
    ax2.axis('off')
    ax3.imshow(np.zeros_like(img_b))
    ax3.text(10,35,str(m1)[:5], color='y', size="25")
    ax3.axis('off')
    plt.show()
