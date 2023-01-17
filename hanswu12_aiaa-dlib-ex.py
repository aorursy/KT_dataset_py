# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install imutils
import dlib, cv2, os
from imutils import face_utils
import numpy as np
import matplotlib.pyplot as plt
import timeit

%matplotlib inline

detector = dlib.get_frontal_face_detector()
img = cv2.imread('/kaggle/input/facerec/dlib/dlib/faces/bald_guys.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.rcParams["figure.figsize"] = (30,15)
plt.imshow(img)
dets, scores, idx = detector.run(img, 1, 0)
dets

for i, d in enumerate(dets):
    if(scores[i]):
        print("Detection {}, score: {}, face_type:{}".format(
                d, scores[i], idx[i]))
        img = cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),(255,0,0),2)

plt.rcParams["figure.figsize"] = (30,15)
plt.imshow(img)

imgd =  dlib.load_rgb_image('/kaggle/input/facerec/dlib/dlib/faces/bald_guys.jpg')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('/kaggle/input/facerec/dlib/dlib/mod/shape_predictor_5_face_landmarks.dat')
dets = detector(imgd, 1)
num_faces = len(dets)
print(num_faces)
faces = dlib.full_object_detections()
for detection in dets:
    faces.append(sp(imgd, detection))
print(len(faces))
images = dlib.get_face_chips(imgd, faces, size=320)
plt.figure(figsize=(20,20)) # specifying the overall grid size
for ind,image in enumerate(images):
    plt.imshow(image)
    plt.subplot(5,5,ind+1)    # the number of images in the grid is 5*5 (25)
    plt.imshow(image)

plt.show()    


image = cv2.imread('/kaggle/input/facerec/dlib/dlib/faces/bald_guys.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor = dlib.shape_predictor('/kaggle/input/facerec/dlib/dlib/mod/shape_predictor_68_face_landmarks.dat')
for k, d in enumerate(dets):
    #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
    #        k, d.left(), d.top(), d.right(), d.bottom()))
    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, d)
    #print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
    #                                              shape.part(1)))
    shape = face_utils.shape_to_np(shape)
    
    (x, y, w, h) = face_utils.rect_to_bb(d)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 255, 0), 3)

#plt.figure(figsize=(10,10))
plt.imshow(image)
plt.xticks([])
plt.yticks([])
plt.show()

imgd =  dlib.load_rgb_image('/kaggle/input/facerec/dlib/dlib/faces/bald_guys.jpg')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('/kaggle/input/facerec/dlib/dlib/mod/shape_predictor_5_face_landmarks.dat')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#cnn_face_detector = dlib.cnn_face_detection_model_v1("/kaggle/input/facerec/dlib/dlib/mod/mmod_human_face_detector.dat")
dets = detector(imgd, 1)
num_faces = len(dets)
print(num_faces)
faces = dlib.full_object_detections()
for detection in dets:
    faces.append(sp(imgd, detection))
print(len(faces))
for k, d in enumerate(dets):
    #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
    #        k, d.left(), d.top(), d.right(), d.bottom()))
    # Get the landmarks/parts for the face in box d.
    shape = sp(imgd, d)
    #print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
    #                                              shape.part(1)))
    shape = face_utils.shape_to_np(shape)
    
    (x, y, w, h) = face_utils.rect_to_bb(d)
    cv2.rectangle(imgd, (x, y), (x + w, y + h), (0, 255, 0), 3)

    for (x, y) in shape:
        cv2.circle(imgd, (x, y), 1, (0, 255, 0), 3)

#plt.figure(figsize=(10,10))
plt.imshow(imgd)
plt.xticks([])
plt.yticks([])
plt.show()

imgd =  dlib.load_rgb_image('/kaggle/input/facerec/dlib/dlib/faces/bald_guys.jpg')
facerec = dlib.face_recognition_model_v1("/kaggle/input/facerec/dlib/dlib/mod/dlib_face_recognition_resnet_model_v1.dat")
shape = sp(imgd, dets[0])
face_descriptor = facerec.compute_face_descriptor(img, shape)
print(face_descriptor)
face_descriptor.shape

faces_folder="/kaggle/input/facerec/dlib/dlib/faces/"
training_xml_path="/kaggle/output"
# Now let's do the training.  The train_simple_object_detector() function has a
# bunch of options, all of which come with reasonable default values.  The next
# few lines goes over some of these options.
options = dlib.simple_object_detector_training_options()
# Since faces are left/right symmetric we can tell the trainer to train a
# symmetric detector.  This helps it get the most value out of the training
# data.
options.add_left_right_image_flips = True
# The trainer is a kind of support vector machine and therefore has the usual
# SVM C parameter.  In general, a bigger C encourages it to fit the training
# data better but might lead to overfitting.  You must find the best C value
# empirically by checking how well the trained detector works on a test set of
# images you haven't trained on.  Don't just leave the value set at 5.  Try a
# few different C values and see what works best for your data.
options.C = 5
# Tell the code how many CPU cores your computer has for the fastest training.
options.num_threads = 4
options.be_verbose = True


training_xml_path = os.path.join(faces_folder, "training.xml")
testing_xml_path = os.path.join(faces_folder, "testing.xml")
# This function does the actual training.  It will save the final detector to
# face.svm.  The input is an XML file that lists the images in the training
# dataset and also contains the positions of the face boxes.  To create your
# own XML files you can use the imglab tool which can be found in the
# tools/imglab folder.  It is a simple graphical tool for labeling objects in
# images with boxes.  To see how to use it read the tools/imglab/README.txt
# file.  But for this example, we just use the training.xml file included with
# dlib.
dlib.train_simple_object_detector(training_xml_path, "face.svm", options)
simple_detector = dlib.simple_object_detector("face.svm")
image = cv2.imread('/kaggle/input/facerec/dlib/dlib/faces/2008_004176.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
simple_dets = simple_detector(image)
for i, d in enumerate(simple_dets):
    if(scores[i]>0):
        image = cv2.rectangle(image,(d.left(),d.top()),(d.right(),d.bottom()),(255,0,0),2)
plt.rcParams["figure.figsize"] = (30,15)
plt.imshow(image)        

from imutils import paths
from scipy.io import loadmat
from skimage import io
image_path="/kaggle/input/facerec/dlib/dlib/train/stop/images/"
anno_path="/kaggle/input/facerec/dlib/dlib/train/stop/anno/"
options = dlib.simple_object_detector_training_options()
options.C = 5
# Tell the code how many CPU cores your computer has for the fastest training.
options.num_threads = 4
options.be_verbose = True

images = [] #存放相片圖檔
boxes = [] #存放Annotations
 
#依序處理path下的每張圖片
for imagePath in paths.list_images(image_path):
#從圖片路徑名稱中取出ImageID
    imageID = imagePath[imagePath.rfind("\\") + 1:].split("_")[1]
    imageID = imageID.replace(".jpg", "")
#載入Annotation
    p = "{}/annotation_{}.mat".format(anno_path, imageID)
    annotations = loadmat(p)["box_coord"]
    print(annotations)
#取出annotations資訊繪成矩形物件，放入boxes變數中。
    bb = [dlib.rectangle(left=x, top=y, right=w, bottom=h) for (y, h, x, w) in annotations]
    boxes.append(bb)
 
#將圖片放入images變數
    images.append(io.imread(imagePath))

#丟入三個參數開始訓練
print("[INFO] training detector...")
detector = dlib.train_simple_object_detector(images, boxes, options)
 
# 將訓練結果匯出到檔案
print("[INFO] dumping classifier to file...")
detector.save("stop_sign.svm")

detector = dlib.simple_object_detector("stop_sign.svm")
image = cv2.imread("/kaggle/input/facerec/dlib/dlib/test/4.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
boxes = detector(image,1)
#print(image)
#在圖片上繪出該矩形
for b in boxes:
    (x, y, w, h) = (b.left(), b.top(), b.right(), b.bottom())
    cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
 
#顯示圖片
plt.imshow(image)

detector = dlib.cnn_face_detection_model_v1('/kaggle/input/facerec/dog/dog/dogHeadDetector.dat')
predictor = dlib.shape_predictor('/kaggle/input/facerec/dog/dog/landmarkDetector.dat')

img_path = '/kaggle/input/facerec/dog/dog/img/05.jpg'
filename, ext = os.path.splitext(os.path.basename(img_path))
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)

plt.figure(figsize=(16, 16))
plt.imshow(img)
dets = detector(img, upsample_num_times=1)

print(dets)

img_result = img.copy()

for i, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

    x1, y1 = d.rect.left(), d.rect.top()
    x2, y2 = d.rect.right(), d.rect.bottom()

    cv2.rectangle(img_result, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=(255,0,0), lineType=cv2.LINE_AA)
    
plt.figure(figsize=(16, 16))
plt.imshow(img_result)
shapes = []

for i, d in enumerate(dets):
    shape = predictor(img, d.rect)
    shape = face_utils.shape_to_np(shape)
    
    for i, p in enumerate(shape):
        shapes.append(shape)
        cv2.circle(img_result, center=tuple(p), radius=3, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(img_result, str(i), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

img_out = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)
cv2.imwrite('img/%s_out%s' % (filename, ext), img_out)
plt.figure(figsize=(16, 16))
plt.imshow(img_result)

from math import atan2, degrees
# overlay function
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    img_to_overlay_t = cv2.cvtColor(img_to_overlay_t, cv2.COLOR_BGRA2RGBA)
    bg_img = background_img.copy()
    # convert 3 channels to 4 channels
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGB2RGBA)

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    b, g, r, a = cv2.split(img_to_overlay_t)

    mask = cv2.medianBlur(a, 5)

    h, w, _ = img_to_overlay_t.shape
    roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

    bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

    # convert 4 channels to 4 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGBA2RGB)

    return bg_img

def angle_between(p1, p2):
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    return degrees(atan2(yDiff, xDiff))
img_result2 = img.copy()

horns = cv2.imread('/kaggle/input/facerec/dog/dog/img/horns2.png',  cv2.IMREAD_UNCHANGED)
horns_h, horns_w = horns.shape[:2]

nose = cv2.imread('/kaggle/input/facerec/dog/dog/img/nose.png',  cv2.IMREAD_UNCHANGED)

for shape in shapes:
    horns_center = np.mean([shape[4], shape[1]], axis=0) // [1, 1.3]
    horns_size = np.linalg.norm(shape[4] - shape[1]) * 3
    
    nose_center = shape[3]
    nose_size = horns_size // 4

    angle = -angle_between(shape[4], shape[1])
    M = cv2.getRotationMatrix2D((horns_w, horns_h), angle, 1)
    rotated_horns = cv2.warpAffine(horns, M, (horns_w, horns_h))

    img_result2 = overlay_transparent(img_result2, nose, nose_center[0], nose_center[1], overlay_size=(int(nose_size), int(nose_size)))
    try:
        img_result2 = overlay_transparent(img_result2, rotated_horns, horns_center[0], horns_center[1], overlay_size=(int(horns_size), int(horns_h * horns_size / horns_w)))
    except:
        print('failed overlay image')

img_out2 = cv2.cvtColor(img_result2, cv2.COLOR_RGB2BGR)
cv2.imwrite('img/%s_out2%s' % (filename, ext), img_out2)
plt.figure(figsize=(16, 16))
plt.imshow(img_result2)

!pip install MTCNN
from mtcnn import MTCNN
import cv2
face_cascade = cv2.CascadeClassifier('/kaggle/input/facerec/haarcascade_frontalface_default.xml')

img_path = '/kaggle/input/facerec/faces.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
start = timeit.default_timer()
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
stop = timeit.default_timer()

print('Time: ', stop - start)  

for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

plt.imshow(img)
detector = MTCNN()
img = cv2.cvtColor(cv2.imread("/kaggle/input/facerec/faces.jpg"), cv2.COLOR_BGR2RGB)

start = timeit.default_timer()
face_rects=detector.detect_faces(img)
stop = timeit.default_timer()

print('Time: ', stop - start)  
for i, d in enumerate(face_rects):
  x1 = d['box'][0]
  y1 = d['box'][1]
  x2 = d['box'][0]+d['box'][2]
  y2 = d['box'][1]+d['box'][3]

  # 以方框標示偵測的人臉
  cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

# 顯示結果
plt.imshow(img)
face_rects
dlib_detector = dlib.get_frontal_face_detector()
img = cv2.cvtColor(cv2.imread("/kaggle/input/facerec/faces.jpg"), cv2.COLOR_BGR2RGB)

start = timeit.default_timer()
face_rects = dlib_detector(img, 0)
stop = timeit.default_timer()

print('Time: ', stop - start)  

for i, d in enumerate(face_rects):
  x1 = d.left()
  y1 = d.top()
  x2 = d.right()
  y2 = d.bottom()

  # 以方框標示偵測的人臉
  cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

# 顯示結果
plt.imshow(img)
cnn_face_detector = dlib.cnn_face_detection_model_v1("/kaggle/input/facerec/dlib/dlib/mod/mmod_human_face_detector.dat")

img = cv2.cvtColor(cv2.imread("/kaggle/input/facerec/faces.jpg"), cv2.COLOR_BGR2RGB)

start = timeit.default_timer()
face_rects = cnn_face_detector(img, 1)
stop = timeit.default_timer()

print('Time: ', stop - start)  

for i, d in enumerate(face_rects):
  x1 = d.rect.left()
  y1 = d.rect.top()
  x2 = d.rect.right()
  y2 = d.rect.bottom()

  # 以方框標示偵測的人臉
  cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

# 顯示結果
plt.imshow(img)

