!pip3 install mtcnn
import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt 

from mtcnn import MTCNN

from PIL import Image

from IPython.display import display 

from matplotlib import cm

import dlib

import cv2

p = '../input/dlibfacelanmarkformaskdetection/shape_predictor_68_face_landmarks.dat'

predictor = dlib.shape_predictor(p)

fdetector = dlib.get_frontal_face_detector()



detector = MTCNN()
path = '../input/test-img-for-mtcnn/__results___9_0.png'

display(Image.open(path))

image  = Image.open(path).convert('RGB')

image_np = np.array(image)

faces = detector.detect_faces(image_np)
print(faces)
path = '../input/face-mask-detection/images/maksssksksss104.png'

display(Image.open(path))

image  = Image.open(path).convert('RGB')

image_np = np.array(image)

faces = detector.detect_faces(image_np)

print(faces)
results = faces[5]

for res in faces: # Going through each face detected

    x_1, y_1, width, height = res['box'] # Getting the box info 

    res_img = image_np[y_1:y_1 + height, x_1:x_1 + width] # croping the face only 

    #print(res_img.shape)

#     new_im = Image.fromarray(res_img)

#     display(new_im)

    #image = cv2.imread(test_img_path)

    plt.imshow(res_img)

    plt.show()

    image = res_img.copy()

    shape = predictor(res_img, dlib.rectangle(int(height), int(0), int(0), int(width)))

    for n in range(0, 68):

        x = shape.part(n).x

        y = shape.part(n).y

        print(x, y)

        cv2.circle(image, (x, y), 1 , (255, 0, 0), -1)

    plt.imshow(image)

    plt.show()
help(predictor)