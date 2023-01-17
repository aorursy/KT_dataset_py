# importing necessary libraries
import cv2
import time
import numpy as np
import tensorflow.keras.backend as K
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input
from tensorflow.keras import optimizers
import tensorflow as tf
# Defining the model architecture.
def create_model():
    K.clear_session()
    ip = Input(shape = (150,150,1))
    z = Conv2D(filters = 32, kernel_size = (64,64), padding='same', input_shape = (150,150,1), activation='relu')(ip)
    z = Conv2D(filters = 64, kernel_size = (16,16), padding='same', input_shape = (150,150,1), activation='relu')(z)
    z = Conv2D(filters = 128, kernel_size = (4,4), padding='same', input_shape = (150,150,1), activation='relu')(z)
    z = MaxPool2D(pool_size = (4,4))(z)
    z = Flatten()(z)
    z = Dense(32, activation='relu')(z)
    op = Dense(6, activation='softmax')(z)
    model = Model(inputs=ip, outputs=op)
    return model

# Loading the pretrained weights.
model = create_model()
model.load_weights('model_weights.h5')
# Detecting the hand using YOLOv3 or HAAR cascade.
def hand_detect(img, type='haar'):

    cv2.imshow('output', img)
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        cv2.destroyAllWindows()

    hand_img = img.copy()
    if type=='yolo':
        width, height, inference_time, results = yolo.inference(img) # getting the hand inferences using YOLOv3.
    elif type=='haar':
        results = cascade.detectMultiScale(hand_img, scaleFactor=1.3, minNeighbors=7) # getting the hand inferences using HAAR cascade.

    if len(results) > 0:
        if type=='yolo':
            _,_,_, x, y, w, h = results[0] # returning the hand location.
            return x,y,150,150
        elif type=='haar':
            x, y, w, h = results[0] # returning the hand location.
            return x-50,y-70,150,150
    else:
        return []
# Bringing all together.
def main():
    roi = []
    # Initial loop for detecting the fist/palm and defining a ROI.
    while(len(roi) == 0):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        roi = hand_detect(frame)

    cv2.destroyAllWindows()
    ret = tracker.init(frame, roi) # initializing the tracker using ROI.
    cv2.destroyAllWindows()

    c = 0
    d = c+1

    while True:
        ret, frame = cap.read() # getting the input frames
        frame = cv2.flip(frame, 1)
        success, roi = tracker.update(frame) # tracking the ROI in the new input frame
        (x,y,w,h) = tuple(map(int, roi)) # defining the ROI box's coordinates.
        if success:
            pt1 = (x,y)
            pt2 = (x+w, y+h)
            square  = frame[y:y+h, x:x+w] # extracting the image(hand gesture) inside ROI.
            gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY) # converting the image to grayscale for contour detection.
            gray_ = cv2.medianBlur(gray, 7) # applying blur to reduce the salt and pepper noise.
            hand = contours(gray, th=150)[0] # converting the grayscale image to binary.
            im = np.array([hand]) # adding a last dimension to the image array to make it model input compatible.
            im = im.reshape(-1,150,150,1) # reshaping the image to be perfectly model compatible.
            result = pred(im) # getting the class label for the image from pretrained model.
            
            cv2.rectangle(frame, pt1, pt2, (255,255,0), 3) # Drawing a rectangle around the detected ROI.
            emo = icon(result) # Getting the emoji corresponding to the predicted class label.

            frame_copy = paste(frame, emo) # Pasting the emoji image over the ROI in input frame
            # Displaying the extracted ROI (what the model sees for prediction) on the top left corner.
            (a,b) = hand.shape
            for i in range(3):
                hand = hand*255
                frame_copy[0:a, 0:b, i] = hand

        # retry the above steps if the tracker fails to detect a hand.
        else:
            cv2.putText(frame, 'Failed to detect object', (100,200), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            cv2.imshow('output', frame_copy)
            roi = []

            while(len(roi) == 0):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                roi = hand_detect(frame)
            continue
        cv2.imshow('output', frame_copy)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
# This function defines the contour of the hand gesture and converts the image to binary.
def contours(diff, th=100):
    _ , thresholded = cv2.threshold(diff, th, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        hand_segment = max(contours, key=cv2.contourArea)
    return (thresholded, hand_segment)
# Predicting the class label of the hand gesture in ROI.
def pred(img):
    return model.predict(img).argmax(axis=1)[0]
# This function returns the file path to the emoji picture.
def icon(x, e=emoji):
    return e[x - 1]
# Pasting the emoji image over the ROI.
def paste(frame, s_img, pt1, pt2):
    x, y = pt1
    x_w, y_h = pt2
    n = min(x_w-x, y_h-y)
    s_img = cv2.resize(s_img, dsize=(n,n))
    l_img = frame
    (x_offset,y_offset,_) = (frame.shape)
    (y_offset,x_offset) = (x_offset//2 - 72, y_offset//2 - 72)
    y1, y2 = y_offset, y_offset + s_img.shape[0]
    x1, x2 = x_offset, x_offset + s_img.shape[1]

    alpha_s = s_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        l_img[y:y_h, x:x_w, c] = (alpha_s * s_img[:, :, c] + alpha_l * l_img[y:y_h, x:x_w, c])
        
    return l_img
# Loading the 5 emoji images into a list
emoji = []
for i in range(1,6):
    img = cv2.imread(f'./emoji/{i}.png', -1)
    emoji.append(img)

# Defining the hand detection methods haar cascads and YOLO.
cascade = cv2.CascadeClassifier('haarcascade/fist.xml')
yolo = YOLO("./models/cross-hands.cfg", "./models/cross-hands.weights", ["hand"])
# Defining multiple tracking objects for tracking the hand.
tk = [cv2.TrackerBoosting_create(), cv2.TrackerMIL_create(), cv2.TrackerKCF_create(), cv2.TrackerTLD_create(), cv2.TrackerCSRT_create()]
tracker = tk[2]
cap = cv2.VideoCapture(0)
main()
