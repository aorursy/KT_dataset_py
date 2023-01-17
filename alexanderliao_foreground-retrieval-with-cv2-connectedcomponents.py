import cv2
import numpy as np
from PIL import Image
img_orig = cv2.imread('../input/train/0b2eb27b5.jpg')[:,:,::-1]
im=Image.fromarray(img_orig).convert("L")
Image.fromarray(img_orig)
img = cv2.threshold(np.array(im), 120, 255, cv2.THRESH_BINARY)[1]  # ensure binary
Image.fromarray(img)
ret, labels = cv2.connectedComponents(img)
def undesired_objects(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    return img2
img2=1-(undesired_objects(img)/255)
img2.shape[0]
img3=img_orig*(img2.reshape(img2.shape[0],img2.shape[1],1))
Image.fromarray(img3.astype(np.uint8))
1-img2
