!pip3 install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl
!wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5
!pip install tensorflow==1.14.0
!pip install keras==2.2.0
import tensorflow as tf
print(tf.__version__)
from imageai.Detection import ObjectDetection
import os
import time

exec_path = os.getcwd()
yolo_obj = ObjectDetection()
yolo_obj.setModelTypeAsYOLOv3()
yolo_obj.setModelPath( os.path.join(exec_path , "yolo.h5"))
yolo_obj.loadModel()
from PIL import Image 
Image.open("../input/object-detect-images/img1.jpg")
start = time.time()
detections = yolo_obj.detectObjectsFromImage(input_image=os.path.join(exec_path , "../input/object-detect-images/img1.jpg"), output_image_path=os.path.join(exec_path , "out_img1.jpg"))
print('Time Taken (in seconds)',time.time() - start)
for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
Image.open("out_img1.jpg")
Image.open("../input/object-detect-images/img2.jpg")
start = time.time()
detections = yolo_obj.detectObjectsFromImage(input_image=os.path.join(exec_path , "../input/object-detect-images/img2.jpg"), output_image_path=os.path.join(exec_path , "out_img2.jpg"))
print('Time Taken (in seconds)',time.time() - start)
for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
Image.open("out_img2.jpg")
Image.open("../input/object-detect-images/img3.jpg")
start = time.time()
detections = yolo_obj.detectObjectsFromImage(input_image=os.path.join(exec_path , "../input/object-detect-images/img3.jpg"), output_image_path=os.path.join(exec_path , "out_img3.jpg"))
print('Time Taken (in seconds)',time.time() - start)
for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
Image.open("out_img3.jpg")
Image.open("../input/object-detect-images/img4.jpg")
start = time.time()
detections = yolo_obj.detectObjectsFromImage(input_image=os.path.join(exec_path , "../input/object-detect-images/img4.jpg"), output_image_path=os.path.join(exec_path , "out_img_4.jpg"))
print('Time Taken (in seconds)',time.time() - start)
for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
Image.open("out_img_4.jpg")