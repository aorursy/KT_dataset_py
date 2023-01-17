!pip install mtcnn
import cv2

import mtcnn

from mtcnn.mtcnn import MTCNN

import matplotlib.pyplot as plt



image_path = "../input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/Chip_Knight/Chip_Knight_0001.jpg"

print(mtcnn.__version__)
def detect_face(image):

    detector = MTCNN()

    bounding_boxes = detector.detect_faces(image)

    return bounding_boxes
def draw_bounding_boxes(image, bboxes):

    for box in bboxes:

        x1, y1, w, h = box['box']

        cv2.rectangle(image, (x1, y1), (x1+w,y1+h), (0,255,0), 2)
def mark_key_point(image, keypoint):

    cv2.circle(image, (keypoint), 1, (0,255,0), 2)
image = cv2.imread(image_path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



bboxes = detect_face(image)

print("Output of MTCNN detector is...\n",bboxes)
# draw bounding box around the detected face and mark facial keypoints

draw_bounding_boxes(image, bboxes)

mark_key_point(image, bboxes[0]['keypoints']['left_eye'])

mark_key_point(image, bboxes[0]['keypoints']['right_eye'])

mark_key_point(image, bboxes[0]['keypoints']['nose'])

mark_key_point(image, bboxes[0]['keypoints']['mouth_left'])

mark_key_point(image, bboxes[0]['keypoints']['mouth_right'])



# display the image

plt.figure(figsize=(10,10))

plt.imshow(image)

plt.xticks([])

plt.yticks([])

plt.show()