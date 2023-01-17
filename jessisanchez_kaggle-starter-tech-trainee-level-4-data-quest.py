# import the necessary packages
import numpy as np
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
# Detector is a pre-trained model to localize a face in a picture
detector = cv2.dnn.readNetFromCaffe("../input/opencv-face-recognizer/deploy.prototxt",\
                                    "../input/opencv-face-recognizer/res10_300x300_ssd_iter_140000.caffemodel")
# Embedder is the pre-trained model to extract face embeddings
embedder = cv2.dnn.readNetFromTorch("../input/opencv-face-recognizer/openface_nn4.small2.v1.t7")

# fill our training set with some photos which do not show you
imagePaths = ["../input/opencv-face-recognizer/images/unknown.jpg",\
              "../input/opencv-face-recognizer/images/unknown2.jpg",\
              "../input/opencv-face-recognizer/images/unknown3.jpg",\
              "../input/opencv-face-recognizer/images/unknown4.jpg",\
              "../input/sqpics/sqPics/selfies/unknown_0.jpg",\
              "../input/sqpics/sqPics/selfies/unknown_1.jpg",\
              "../input/sqpics/sqPics/selfies/unknown_2.jpg",\
              "../input/sqpics/sqPics/selfies/unknown_3.jpg",\
              "../input/sqpics/sqPics/selfies/unknown_4.jpg"]

# now add the file names of our input photos
for root, dirs, files in os.walk("../input/sqpics/sqPics/JessiSq"):
    for file in files:
        if file.endswith((".jpg", ".jpeg", ".png", ".bmp")):
            imagePaths.append(os.path.join(root, file))

# YOUR turn: insert your name here in this variable
name = "Jessi"

# initialize our lists of extracted facial embeddings and
# corresponding people names
knownEmbeddings = []
knownNames = []

images_count = 0

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
# load the image, resize it to have a width of 600 pixels (while
# maintaining the aspect ratio), and then grab the image dimensions
    image = cv2.imread(imagePath)
    new_width = 600
    new_height = (image.shape[0] / image.shape[1]) * new_width 
    image = cv2.resize(image, (int(new_width), int(new_height)))
    (h, w) = image.shape[:2]
    image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300,300),(104, 177, 123), swapRB=False, crop=False) ## potential root cause ?
                       
# apply OpenCV's deep learning-based face detector to localize faces in the input image
    detector.setInput(image_blob)
    detections = detector.forward()

    # ensure at least one face was found
    if len(detections) > 0:
        # we're making the assumption that each image has only ONE face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # ensure that the detection with the largest probability also means our minimum probability test 
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI and grab the ROI dimensions
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob through our face embedding model to 
            # obtain the 128-d quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # add the name of the person and corresponding face embedding to their respective lists
            if images_count>= 4:
                knownNames.append(name)
            else:
                knownNames.append("Unknown person")
            knownEmbeddings.append(vec.flatten())
            images_count +=1
            
# finally store the embeddings in a dictionary
data = {"embeddings": knownEmbeddings, "names": knownNames}

print(str(images_count-4) + " of your images used for training")

#create a label encoder (a pre-trained model) that contains the names to be learned for the faces
le = LabelEncoder()

# train the label encoder
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face
recognizer = SVC(C=1.0, kernel="linear", probability=True)

#here goes your code to train the model
recognizer.fit(data["embeddings"], labels)
#YOUR turn: set the path to your image
image_path = "../input/sqjessandfriends/JessiAndFriends.jpg"
# image_path = "../input/jessiandfriends/imagesWithJessi/JessiAndFriends.jpg"
# image_path = "../input/jessiandfriends/imagesWithJessi/JessiAndK.jpg"
# image_path = "../input/jessiandfriends/imagesWithJessi/YoungJessiAndMum.jpg"
# image_path = "../input/mypictures/dataset/Jessi/Jessi1.jpg" 
# image_path = "../input/mypictures/dataset/Jessi/Jessi2.jpg"
# image_path = "../input/mypictures/dataset/Jessi/Jessi3.jpg"
# image_path = "../input/mypictures/dataset/Jessi/Jessi4.jpg"

# load the image, resize it to have a width of 600 pixels (while
# maintaining the aspect ratio), and then grab the image dimensions
image_src = cv2.imread(image_path)
new_width = 600
new_height = (image_src.shape[0] / image_src.shape[1]) * new_width 
image = cv2.resize(image_src, (int(new_width), int(new_height)))
(h, w) = image.shape[:2]

# construct a blob from the image
image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)

# YOUR turn:  same as for training - apply OpenCV's deep learning-based face detector to localize faces in the input image
# your code goes here

# loop over all detected faces
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the prediction
    confidence = detections[0, 0, i, 2]

    # continue only for faces with a sufficient confidence, e.g. 50%
    if confidence > 0.5:
        #compute the (x, y)-coordinates of the bounding box for the face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # extract the face ROI
        face = image[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]

        # ensure the face width and height are sufficiently large
        if fW < 20 or fH < 20:
            continue

        # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d
        # quantification of the face
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),(0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vec = embedder.forward()

        # YOUR turn: insert the code to use our recognizer to perform the classification for the just created vector (=face data)
        # and return an array of probabilitesi
        preds = recognizer.predict_proba(vec)[0] #...
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        # draw the bounding box of the face along with the associated probability
        text = "{}: {:.2f}%".format(name, proba * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
cv2.imwrite("result.jpg", image)

from IPython.display import Image
Image("result.jpg") 
