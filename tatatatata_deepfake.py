# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import file utilities
import os
import glob

# import charting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation 
%matplotlib inline

from IPython.display import HTML

# import computer vision
import cv2

TEST_PATH = '../input/deepfake-detection-challenge/test_videos/'
TRAIN_PATH = '../input/deepfake-detection-challenge/train_sample_videos/'

metadata = '../input/deepfake-detection-challenge/train_sample_videos/metadata.json'
# load the filenames for train videos
train_fns = sorted(glob.glob(TRAIN_PATH + '*.mp4'))
# load the filenames for test videos
test_fns = sorted(glob.glob(TEST_PATH + '*.mp4'))

print('There are {} samples in the train set.'.format(len(train_fns)))
print('There are {} samples in the test set.'.format(len(test_fns)))

meta = pd.read_json(metadata).transpose()
print(meta.head())
print(meta.describe())

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'FAKE', 'REAL'
sizes = [meta[meta.label == 'FAKE'].label.count(), meta[meta.label == 'REAL'].label.count()]

fig1, ax1 = plt.subplots(figsize=(10,7))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors=['#f4d53f', '#02a1d8'])
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Labels', fontsize=16)

plt.show()
def get_frame(filename):
    '''
    Helper function to return the 1st frame of the video by filename
    INPUT: 
        filename - the filename of the video
    OUTPUT:
        image - 1st frame of the video (RGB)
    '''
    # Playing video from file
    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()

    # Our operations on the frame come here
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    return image

def get_label(filename, meta):
    '''
    Helper function to get a label from the filepath.
    INPUT:
        filename - filename of the video
        meta - dataframe containing metadata.json
    OUTPUT:
        label - label of the video 'FAKE' or 'REAL'
    '''
    video_id = filename.split('/')[-1]
    return meta.loc[video_id].label

def get_original_filename(filename, meta):
    '''
    Helper function to get the filename of the original image
    INPUT:
        filename - filename of the video
        meta - dataframe containing metadata.json
    OUTPUT:
        original_filename - name of the original video
    '''
    video_id = filename.split('/')[-1]
    original_id = meta.loc[video_id].original
    
    return original_id

def visualize_frame(filename, meta, train = True):
    '''
    Helper function to visualize the 1st frame of the video by filename and metadata
    INPUT:
        filename - video filename
        meta - dataframe containing metadata.json
        train - indicates that the video is among train samples and the label can be retrived from metadata
    '''
    # get the 1st frame of the video
    image = get_frame(filename)

    # Display the 1st frame of the video
    fig, axs = plt.subplots(1,3, figsize=(20,7))
    axs[0].imshow(image) 
    axs[0].axis('off')
    axs[0].set_title('Original frame')
    
    # Extract the face with haar cascades
    face_cascade = cv2.CascadeClassifier('../input/haarcascades/haarcascade_frontalface_default.xml')

    # run the detector
    # the output here is an array of detections; the corners of each detection box
    # if necessary, modify these parameters until you successfully identify every face in a given image
    faces = face_cascade.detectMultiScale(image, 1.2, 3)

    # make a copy of the original image to plot detections on
    image_with_detections = image.copy()

    # loop over the detected faces, mark the image where each face is found
    for (x,y,w,h) in faces:
        # draw a rectangle around each detected face
        # you may also need to change the width of the rectangle drawn depending on image resolution
        cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3)

    axs[1].imshow(image_with_detections)
    axs[1].axis('off')
    axs[1].set_title('Highlight faces')
    
    # crop out the 1st face
    crop_img = image.copy()
    for (x,y,w,h) in faces:
        crop_img = image[y:y+h, x:x+w]
        break;
        
    # plot the 1st face
    axs[2].imshow(crop_img)
    axs[2].axis('off')
    axs[2].set_title('Zoom-in face')
    
    if train:
        plt.suptitle('Image {image} label: {label}'.format(image = filename.split('/')[-1], label=get_label(filename, meta)))
    else:
        plt.suptitle('Image {image}'.format(image = filename.split('/')[-1]))
    plt.show()

print(meta.loc[meta.label=='FAKE'].describe)
fake_train_sample_video = list(meta.loc[meta.label=='FAKE'].sample(10).index)
print(fake_train_sample_video)

def display_image_from_video(video_path):
    '''
    input: video_path - path for video
    process:
    1. perform a video capture from the video
    2. read the image
    3. display the image
    '''
    capture_image = cv2.VideoCapture(video_path) 
    ret, frame = capture_image.read()
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.imshow(frame)
    
DATA_FOLDER = '../input/deepfake-detection-challenge'
TRAIN_SAMPLE_FOLDER = 'train_sample_videos'
for video_file in fake_train_sample_video:
    print(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video_file))


img_list = ['../input/deepfake-detection-challenge/train_sample_videos/abarnvbtwb.mp4', 
'../input/deepfake-detection-challenge/train_sample_videos/aelfnikyqj.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/afoovlsmtx.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/agrmhtjdlk.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/ahqqqilsxt.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/awhmfnnjih.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/cwbacdwrzo.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/cxttmymlbn.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/eprybmbpba.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/bbvgxeczei.mp4']

i = 0
while i < len(img_list):
    visualize_frame(img_list[i], meta)
    i += 1
img_list = ['../input/deepfake-detection-challenge/train_sample_videos/abarnvbtwb.mp4', 
'../input/deepfake-detection-challenge/train_sample_videos/aelfnikyqj.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/afoovlsmtx.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/agrmhtjdlk.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/ahqqqilsxt.mp4']
i = 0
while i < len(img_list):
    visualize_frame(img_list[i], meta)
    i += 1

fake_train_sample_video = list(meta.loc[meta.label=='REAL'].sample(5).index)
fake_train_sample_video
def display_image_from_video(video_path):
    '''
    input: video_path - path for video
    process:
    1. perform a video capture from the video
    2. read the image
    3. display the image
    '''
    capture_image = cv2.VideoCapture(video_path) 
    ret, frame = capture_image.read()
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.imshow(frame)
    
DATA_FOLDER = '../input/deepfake-detection-challenge'
TRAIN_SAMPLE_FOLDER = 'train_sample_videos'
for video_file in fake_train_sample_video:
    display_image_from_video(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video_file))


!pip install dlib
import cv2,matplotlib.pyplot as plt,dlib
detector = dlib.get_frontal_face_detector()
color_green = (0,255,0)
line_width = 3
from PIL import Image
from PIL import ImageFilter


def face_extraction(filename, meta, train = True):
    '''
    Helper function to visualize the 1st frame of the video by filename and metadata
    INPUT:
        filename - video filename
        meta - dataframe containing metadata.json
        train - indicates that the video is among train samples and the label can be retrived from metadata
    '''
    # get the 1st frame of the video
    image = get_frame(filename)
    

    # Display the 1st frame of the video
    fig, axs = plt.subplots(1,2, figsize=(20,7))
    axs[0].imshow(image) 
    axs[0].axis('off')
    axs[0].set_title('Original frame')
    
    # Extract the face with haar cascades
#    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dets = detector(image)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(image, 1)
    print(dets)
    for det in dets:
        cv2.rectangle(image,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
        if len(cv2.rectangle(image,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)) > 0:
            roi = image[det.top():det.bottom(),det.left():det.right()]
            axs[1].imshow(roi)
            axs[1].axis('off')
            axs[1].set_title('Highlight faces')
        else:
            pass
    

    plt.show()

img_list = ['../input/deepfake-detection-challenge/train_sample_videos/abarnvbtwb.mp4', 
'../input/deepfake-detection-challenge/train_sample_videos/aelfnikyqj.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/afoovlsmtx.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/agrmhtjdlk.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/ahqqqilsxt.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/awhmfnnjih.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/cwbacdwrzo.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/cxttmymlbn.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/eprybmbpba.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/bbvgxeczei.mp4']


i = 0
while i < len(img_list):
    face_extraction(img_list[i], meta)
    i += 1

!pip install imutils
import cv2
import numpy as np
from PIL import Image
import dlib
img_list = ['../input/deepfake-detection-challenge/train_sample_videos/abarnvbtwb.mp4', 
'../input/deepfake-detection-challenge/train_sample_videos/aelfnikyqj.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/afoovlsmtx.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/agrmhtjdlk.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/ahqqqilsxt.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/awhmfnnjih.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/cwbacdwrzo.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/cxttmymlbn.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/eprybmbpba.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/bbvgxeczei.mp4']

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../input/dlibpackage/shape_predictor_68_face_landmarks.dat")


i = 0
while i < len(img_list):
    face_extraction(img_list[i], meta)
    i += 1
def face_extraction(filename, meta, train = True):
    '''
    Helper function to visualize the 1st frame of the video by filename and metadata
    INPUT:
        filename - video filename
        meta - dataframe containing metadata.json
        train - indicates that the video is among train samples and the label can be retrived from metadata
    '''
    # get the 1st frame of the video
    image = get_frame(filename)
#    image = imutils.resize(image, width=800)
#    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Display the 1st frame of the video
    fig, axs = plt.subplots(1,2, figsize=(20,7))
    axs[0].imshow(image) 
    axs[0].axis('off')
    axs[0].set_title('Original frame')
    
    # Extract the face with haar cascades
#    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dets = detector(image)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(image, 1)
    print(dets)
    for det in dets:
        cv2.rectangle(image,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
        if len(cv2.rectangle(image,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)) > 0:
            roi = image[det.top():det.bottom(),det.left():det.right()]
            axs[1].imshow(roi)
            axs[1].axis('off')
            axs[1].set_title('Highlight faces')
        else:
            pass
    

    plt.show()

img_list = ['../input/deepfake-detection-challenge/train_sample_videos/abarnvbtwb.mp4', 
'../input/deepfake-detection-challenge/train_sample_videos/aelfnikyqj.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/afoovlsmtx.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/agrmhtjdlk.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/ahqqqilsxt.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/awhmfnnjih.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/cwbacdwrzo.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/cxttmymlbn.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/eprybmbpba.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/bbvgxeczei.mp4']


i = 0
while i < len(img_list):
    face_extraction(img_list[i], meta)
    i += 1
#measure exposure
def exposure(image):
    hist = cv2.calcHist([image],[0],None,[256],[0,256])
    hist1 = hist[0:64]
    hist2 = hist[64:128]
    hist3 = hist[128:192]
    hist4 = hist[192:256]
    plt.hist(image.ravel(),256,[0,256]); 
#measure noise
from skimage.restoration import estimate_sigma

def estimate_noise(image):
    return estimate_sigma(image, multichannel=True, average_sigmas=True)
  
#measure blur
def estimate_blur(image):
    threshold = 80
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(image, cv2.CV_64F).var()
    if score > threshold:
        print("Not Blur")
    else:
        print("Blur")
    
    
from scipy import fftpack
from matplotlib.colors import LogNorm

# Show the results


def plot_spectrum(image):
    im_fft = fftpack.fft2(image)
    plt.imshow(im_fft)
    plt.show()


def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def face_extraction_features_extraction(filename, meta, train = True):
    '''
    Helper function to visualize the 1st frame of the video by filename and metadata
    INPUT:
        filename - video filename
        meta - dataframe containing metadata.json
        train - indicates that the video is among train samples and the label can be retrived from metadata
    '''
    # get the 1st frame of the video
    image = get_frame(filename)
    #image = white_balance(image)

    # Display the 1st frame of the video
    fig, axs = plt.subplots(1,2, figsize=(20,7))
    axs[0].imshow(image) 
    axs[0].axis('off')
    axs[0].set_title('Original frame')
    
    # Extract the face with haar cascades
#    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dets = detector(image)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(image, 1)
    print(dets)
    for det in dets:
        cv2.rectangle(image,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
        if len(cv2.rectangle(image,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)) > 0:
            roi = image[det.top():det.bottom(),det.left():det.right()]
            print('estimate noise: ',estimate_noise(roi))
            print('estimate blur: ',estimate_blur(roi))
            axs[1].imshow(roi)
            axs[1].axis('off')
            axs[1].set_title('Highlight faces')
            plt.show()
            print('exposure')
            exposure(roi)
            plt.show()
        else:
            pass
    

    plt.show()

img_list = ['../input/deepfake-detection-challenge/train_sample_videos/abarnvbtwb.mp4', 
'../input/deepfake-detection-challenge/train_sample_videos/aelfnikyqj.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/afoovlsmtx.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/agrmhtjdlk.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/ahqqqilsxt.mp4']


i = 0
while i < len(img_list):
    face_extraction_features_extraction(img_list[i], meta)
    i += 1

img_list = ['../input/deepfake-detection-challenge/train_sample_videos/awhmfnnjih.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/cwbacdwrzo.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/cxttmymlbn.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/eprybmbpba.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/bbvgxeczei.mp4']


i = 0
while i < len(img_list):
    image = get_frame(img_list[i])
    new_image = cv2.Laplacian(image,cv2.CV_64F)
    plt.figure(figsize=(11,6))
    plt.subplot(131), plt.imshow(image, cmap='gray'),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(new_image, cmap='gray'),plt.title('Laplacian')
    plt.xticks([]), plt.yticks([])
    i += 1
img_list = ['../input/deepfake-detection-challenge/train_sample_videos/awhmfnnjih.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/cwbacdwrzo.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/cxttmymlbn.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/eprybmbpba.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/bbvgxeczei.mp4']

i = 0
while i < len(img_list):
    image = get_frame(img_list[i])
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(image),flags = cv2.DFT_COMPLEX_OUTPUT)
# shift the zero-frequncy component to the center of the spectrum
    dft_shift = np.fft.fftshift(dft)
# save image of the image in the fourier domain.
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    plt.figure(figsize=(11,6))
    plt.subplot(121),plt.imshow(image, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    i += 1
img_list = ['../input/deepfake-detection-challenge/train_sample_videos/awhmfnnjih.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/cwbacdwrzo.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/cxttmymlbn.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/eprybmbpba.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/bbvgxeczei.mp4']


i = 0
while i < len(img_list):
    print(img_list[i])
    image = get_frame(img_list[i])
    new_image = cv2.Laplacian(image,cv2.CV_64F)
    plt.figure(figsize=(11,6))
    plt.subplot(131), plt.imshow(image, cmap='gray'),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(new_image, cmap='gray'),plt.title('Laplacian')
    plt.xticks([]), plt.yticks([])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(image),flags = cv2.DFT_COMPLEX_OUTPUT)
# shift the zero-frequncy component to the center of the spectrum
    dft_shift = np.fft.fftshift(dft)
# save image of the image in the fourier domain.
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    plt.subplot(133),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    
    i += 1
img_list = ['../input/deepfake-detection-challenge/train_sample_videos/abarnvbtwb.mp4', 
'../input/deepfake-detection-challenge/train_sample_videos/aelfnikyqj.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/afoovlsmtx.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/agrmhtjdlk.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/ahqqqilsxt.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/awhmfnnjih.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/cwbacdwrzo.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/cxttmymlbn.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/eprybmbpba.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/bbvgxeczei.mp4']


i = 0
while i < len(img_list):
        # get the 1st frame of the video
    image = get_frame(img_list[i])
    image = white_balance(image)
    # Display the 1st frame of the video
    fig, axs = plt.subplots(1,2, figsize=(20,7))
    
    # Extract the face with haar cascades
#    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dets = detector(image)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.Q
    dets = detector(image, 1)
    for det in dets:
        cv2.rectangle(image,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
        if len(cv2.rectangle(image,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)) == 1080:
            roi = image[det.top():det.bottom(),det.left():det.right()]            
            image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            new_image = cv2.Laplacian(image,cv2.CV_64F)
            plt.figure(figsize=(11,6))
            plt.subplot(131), plt.imshow(roi, cmap='gray'),plt.title('Original')
            plt.xticks([]), plt.yticks([])
            plt.subplot(132), plt.imshow(new_image, cmap='gray'),plt.title('Laplacian')
            plt.xticks([]), plt.yticks([])
            dft = cv2.dft(np.float32(image),flags = cv2.DFT_COMPLEX_OUTPUT)
# shift the zero-frequncy component to the center of the spectrum
            dft_shift = np.fft.fftshift(dft)
# save image of the image in the fourier domain.
            magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
            print('magnitude shape: ',magnitude_spectrum.shape)
            plt.subplot(133),plt.imshow(magnitude_spectrum, cmap = 'gray')
            plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
            plt.show()
 
    i += 1
from PIL import Image, ImageEnhance 

def feature_extraction(filename, meta, train = True):
    '''
    Helper function to visualize the 1st frame of the video by filename and metadata
    INPUT:
        filename - video filename
        meta - dataframe containing metadata.json
        train - indicates that the video is among train samples and the label can be retrived from metadata
    '''
    # get the 1st frame of the video
    image = get_frame(filename)
    image = white_balance(image)

    # Display the 1st frame of the video
    fig, axs = plt.subplots(1,2, figsize=(20,7))
    
    # Extract the face with haar cascades
#    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dets = detector(image)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(image, 1)
    for det in dets:
        cv2.rectangle(image,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
        if len(cv2.rectangle(image,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)) == 1080:
            roi = image[det.top():det.bottom(),det.left():det.right()]            
            #image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            new_image = cv2.Laplacian(roi,cv2.CV_64F)
            f = np.fft.fft2(new_image)
            f_shift = np.fft.fftshift(f)
            f_complex = f_shift
            f_abs = np.abs(f_complex) + 1 # lie between 1 and 1e6
            f_bounded = 20 * np.log(f_abs)
            f_img = 255 * f_bounded / np.max(f_bounded)
            f_img = f_img.astype(np.uint8)
            
            #new_image2 = cv2.Laplacian(image,cv2.CV_64F)
            #new_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            print('estimate noise: ',estimate_noise(roi))
            print('estimate blur: ',estimate_blur(roi))
            print('exposure')
            exposure(roi)
            plt.show()
            plt.figure(figsize=(11,6))
            plt.subplot(131), plt.imshow(roi, cmap='gray'),plt.title('Original')
            plt.xticks([]), plt.yticks([])
            plt.subplot(132), plt.imshow(new_image, cmap='gray'),plt.title('Laplacian')
            #plt.subplot(132), plt.imshow(new_image2, cmap='gray'),plt.title('Laplacian')
            plt.xticks([]), plt.yticks([])
            plt.subplot(133), plt.imshow(f_img, cmap='gray'),plt.title('Spectrum')
            plt.xticks([]), plt.yticks([])
            
            #dft = cv2.dft(np.float32(new_image1),flags = cv2.DFT_COMPLEX_OUTPUT)
# shift the zero-frequncy component to the center of the spectrum
            #dft_shift = np.fft.fftshift(dft)
# save image of the image in the fourier domain.
            #magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
            #plt.subplot(133),plt.imshow(magnitude_spectrum, cmap = 'gray')
            #plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
            #plt.show()
        else:
            pass
        
    

    plt.show()

img_list = ['../input/deepfake-detection-challenge/train_sample_videos/awhmfnnjih.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/cwbacdwrzo.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/cxttmymlbn.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/eprybmbpba.mp4',
'../input/deepfake-detection-challenge/train_sample_videos/bbvgxeczei.mp4']


i = 0
while i < len(img_list):
    print(img_list[i])
    feature_extraction(img_list[i], meta)
    i += 1

img_list = []

print(meta.loc[meta.label=='FAKE'].describe)
fake_train_sample_video = list(meta.loc[meta.label=='FAKE'].index)
i = 0
while i < len(fake_train_sample_video):
    img_list.append('../input/deepfake-detection-challenge/train_sample_videos/'+str(fake_train_sample_video[i]))
    i += 1

i = 0
while i < 10:
    print(img_list[i])
    feature_extraction(img_list[i], meta)
    i += 1
from PIL import Image, ImageEnhance 

def data_preparation(path,filename, meta, train = True):
    '''
    Helper function to visualize the 1st frame of the video by filename and metadata
    INPUT:
        filename - video filename
        meta - dataframe containing metadata.json
        train - indicates that the video is among train samples and the label can be retrived from metadata
    '''
    # get the 1st frame of the video
    image = get_frame(path+filename)
    image = white_balance(image)
    # Display the 1st frame of the video    
    # Extract the face with haar cascades
#    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dets = detector(image)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(image, 1)
    for det in dets:
        cv2.rectangle(image,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
        try:
            if len(cv2.rectangle(image,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)) == 1080:
                roi = image[det.top():det.bottom(),det.left():det.right()]    
            #image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                new_image = cv2.Laplacian(roi,cv2.CV_64F)
                f = np.fft.fft2(new_image)
                f_shift = np.fft.fftshift(f)
                f_complex = f_shift
                f_abs = np.abs(f_complex) + 1 # lie between 1 and 1e6
                f_bounded = 20 * np.log(f_abs)
                f_img = 255 * f_bounded / np.max(f_bounded)
                f_img = f_img.astype(np.uint8)
                return(f_img)
        except:
            pass


real_train_sample_video = list(meta.loc[meta.label=='FAKE'].index)
print(len(real_train_sample_video))
fake_train_sample_video = list(meta.loc[meta.label=='REAL'].index)
print(len(fake_train_sample_video))
path = '../input/deepfake-detection-challenge/train_sample_videos/'

i = 0
while i < len(real_train_sample_video[0:300]):
    print(real_train_sample_video[i])
    laplacian_answer = data_preparation(path,real_train_sample_video[i], meta)
    try:
        cv2.imwrite('train_real_'+str(i)+'.jpg', laplacian_answer) 
        plt.imshow(laplacian_answer)
        plt.show()
    except:
        pass
    i += 1
    
i = 0
while i < len(real_train_sample_video[301:323]):
    print(real_train_sample_video[i])
    laplacian_answer = data_preparation(path,real_train_sample_video[i], meta)
    try:
        cv2.imwrite('test_real_'+str(i)+'.jpg', laplacian_answer)    
        plt.imshow(laplacian_answer)
        plt.show()
    except:
        pass
    i += 1
    
i = 0
while i < len(fake_train_sample_video):
    print(fake_train_sample_video[i])
    laplacian_answer = data_preparation(path,fake_train_sample_video[i], meta)
    try:
        cv2.imwrite('test_fake_'+str(i)+'.jpg', laplacian_answer)
        plt.imshow(laplacian_answer)
        plt.show()
    except:
        pass
    i += 1
print(os.listdir())
import os
data = os.listdir()
train_pictures = []
test_pictures = []
labels = []
data_type = []

i = 0
while i < len(data):
    if data[i][0:5] == 'train':
        train_pictures.append(data[i])
        labels.append(0)
        data_type.append('TRAIN')
    elif data[i][0:4] == 'test':
        test_pictures.append(data[i])
        if data[i][5:9] == 'fake':
            labels.append(0)
            data_type.append('TEST')
        elif data[i][5:9] == 'real':
            labels.append(1)
            data_type.append('TEST')
    else:
        pass
    i += 1

print(len(train_pictures))
print(test_pictures[1])
print(test_pictures[1][0:5])
print(test_pictures)
from PIL import Image, ImageFile
from matplotlib.pyplot import imshow
import requests
import numpy as np
from io import BytesIO

def make_square(img):
    cols,rows = img.size
    
    if rows>cols:
        pad = (rows-cols)/2
        img = img.crop((pad,0,cols,cols))
    else:
        pad = (cols-rows)/2
        img = img.crop((0,pad,rows,rows))
    
    return img
    
x = [] 
dsize = (128, 128)

for pic in train_pictures:
    img = cv2.imread(pic)
    img = cv2.resize(img, dsize)
    plt.imshow(img)
    img_array = np.asarray(img)
    img_array = img_array.flatten()
    img_array = img_array.astype(np.float32)
    img_array = (img_array-128)/128
    x.append(img_array)
    

x = np.array(x)
print(x.shape)
x = []    
y = []
loaded_images = []
    
for pic in train_pictures:
    
    img = cv2.imread(pic)
    img = cv2.resize(img, dsize, Image.ANTIALIAS)
    y.append(img)
    plt.imshow(img)

    img_array = np.asarray(img)
    img_array = img_array.flatten()
    img_array = img_array.astype(np.float32)
    img_array = (img_array-128)/128
    x.append(img_array)
    
    

x = np.array(x)

print(x.shape)




%matplotlib inline
from PIL import Image, ImageFile
from matplotlib.pyplot import imshow
import requests
from io import BytesIO
from sklearn import metrics
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import display, HTML
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping

# Fit reg#ression DNN model.
print("Creating/Training neural network")
model = Sequential()
model.add(Dense(100, input_dim=x.shape[1], activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(x.shape[1])) # Multiple output neurons
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x,x,verbose=1,epochs=10000)



print("Neural network trained")


print(model.summary())
test_pictures = ['test_real_6.jpg','test_real_17.jpg','test_real_0.jpg',  'test_real_11.jpg', 'test_real_20.jpg', 'test_real_4.jpg', 'test_real_8.jpg', 'test_real_3.jpg', 'test_real_16.jpg', 'test_real_21.jpg', 'test_real_13.jpg',  'test_real_10.jpg', 'test_real_15.jpg',  'test_real_18.jpg', 'test_real_12.jpg','test_real_19.jpg',  'test_real_14.jpg','test_real_5.jpg']
img = y[0]
img = cv2.imread(pic)
img = cv2.resize(img, dsize)
plt.imshow(img)
img_array = np.asarray(img)
print(img.size)

#Display noisy image
img2 = img_array.astype(np.uint8)
img2 = Image.fromarray(img2, 'RGB')
plt.imshow(img2)
plt.show()
# Present noisy image to auto encoder
img_array = img_array.flatten()
img_array = img_array.astype(np.float32)
img_array = (img_array-128)/128
img_array = np.array([img_array])
pred = model.predict(img_array)[0]

# Display neural result
img_array2 = pred.reshape(128,128,3)
img_array2 = (img_array2*128)+128
img_array2 = img_array2.astype(np.uint8)
img2 = Image.fromarray(img_array2, 'RGB')

plt.imshow(img2)
plt.show()
y_fake = []

    
for pic in test_pictures:
    img = cv2.imread(pic)
    img = cv2.resize(img, dsize, Image.ANTIALIAS)
    y_fake.append(img)

print(len(y_fake))
print(test_pictures)
i = 0
tt = []
tt2 = []
while i <len(test_pictures):
    img = y_fake[i]
    print(test_pictures[i])
    img = cv2.imread(test_pictures[i])
    img = cv2.resize(img, dsize)
    plt.imshow(img)
    img_array = np.asarray(img)
    print(img.size)

    #Display noisy image
    img2 = img_array.astype(np.uint8)
    img2 = Image.fromarray(img2, 'RGB')
    plt.imshow(img2)
    plt.show()
    # Present noisy image to auto encoder
    img_array = img_array.flatten()
    img_array = img_array.astype(np.float32)
    img_array = (img_array-128)/128
    img_array = np.array([img_array])
    pred = model.predict(img_array)[0]

    # Display neural result
    img_array2 = pred.reshape(128,128,3)
    img_array2 = (img_array2*128)+128
    img_array2 = img_array2.astype(np.uint8)
    img2 = Image.fromarray(img_array2, 'RGB')

    plt.imshow(img2)
    plt.show()
    
    score1 = np.sqrt(metrics.mean_squared_error(pred,img_array[0]))
    print(f"Out of Sample Score (RMSE): {score1}")   
    tt.append(score1)
    tt2.append(score1)
    print('=====================')
    print('=====================')
    i += 1
print(tt2)
data = {'col_1': tt2}
df = pd.DataFrame.from_dict(data)
print(df.describe())

fig1, ax1 = plt.subplots()
ax1.set_title('box plot')
ax1.boxplot(tt2)
plt.scatter(sorted(tt2),range(0,len(tt2)), label='skitscat', color='k', s=25, marker="o")
plt.show()
threshold = 0.041602#0.036230 + 2*0.007128
import math
i = 0
tt = [0.042558003, 0.030652788, 0.042754516, 0.038435463, 0.038629595, 0.041084316, 0.035615653, 0.04272078, 0.033939373, 0.04175356, 0.041320685, 0.0368206, 0.042760883, 0.039931614, 0.03664131, 0.031103585, 0.038762517, 0.038499016]
print(max(tt))
mean_value = sum(tt)/len(tt)
print(mean_value)
while i < len(tt):
    tt[i] = (tt[i]-mean_value)**2
    i += 1
value_std = math.sqrt(sum(tt)/len(tt))
print(value_std)

#mean: 0.03855468094444445
#std: 0.003723231593256656



test_pictures = ['test_fake_23.jpg', 'test_fake_49.jpg', 'test_fake_71.jpg', 'test_fake_41.jpg', 'test_fake_50.jpg', 'test_fake_11.jpg', 'test_fake_27.jpg', 'test_fake_44.jpg', 'test_fake_16.jpg', 'test_fake_30.jpg', 'test_fake_68.jpg', 'test_fake_59.jpg', 'test_fake_70.jpg', 'test_fake_67.jpg', 'test_fake_38.jpg', 'test_fake_55.jpg', 'test_fake_43.jpg', 'test_fake_1.jpg', 'test_fake_22.jpg', 'test_fake_32.jpg', 'test_fake_75.jpg', 'test_fake_63.jpg', 'test_fake_12.jpg', 'test_fake_29.jpg', 'test_fake_8.jpg', 'test_fake_7.jpg', 'test_fake_18.jpg', 'test_fake_0.jpg', 'test_fake_62.jpg', 'test_fake_46.jpg','test_fake_51.jpg', 'test_fake_73.jpg','test_fake_20.jpg', 'test_fake_48.jpg', 'test_fake_4.jpg', 'test_fake_34.jpg','test_fake_58.jpg', 'test_fake_24.jpg', 'test_fake_57.jpg', 'test_fake_5.jpg', 'test_fake_74.jpg', 'test_fake_61.jpg', 'test_fake_15.jpg', 'test_fake_25.jpg', 'test_fake_13.jpg', 'test_fake_33.jpg', 'test_fake_66.jpg', 'test_fake_37.jpg', 'test_fake_65.jpg', 'test_fake_40.jpg', 'test_fake_28.jpg', 'test_fake_64.jpg', 'test_fake_39.jpg', 'test_fake_10.jpg', 'test_fake_35.jpg', 'test_fake_14.jpg', 'test_fake_69.jpg', 'test_fake_26.jpg', 'test_fake_42.jpg', 'test_fake_72.jpg', 'test_fake_47.jpg', 'test_fake_56.jpg', 'test_fake_19.jpg', 'test_fake_36.jpg', 'test_fake_6.jpg', 'test_fake_60.jpg', 'test_fake_54.jpg', 'test_fake_31.jpg']
y_fake = []

    
for pic in test_pictures:
    img = cv2.imread(pic)
    img = cv2.resize(img, dsize, Image.ANTIALIAS)
    y_fake.append(img)

print(len(y_fake))
print(test_pictures)
i = 0
tt = []
tt2 = []
while i <len(test_pictures):
    img = y_fake[i]
    print(test_pictures[i])
    img = cv2.imread(test_pictures[i])
    img = cv2.resize(img, dsize)
    plt.imshow(img)
    img_array = np.asarray(img)
    print(img.size)

    #Display noisy image
    img2 = img_array.astype(np.uint8)
    img2 = Image.fromarray(img2, 'RGB')
    plt.imshow(img2)
    plt.show()
    # Present noisy image to auto encoder
    img_array = img_array.flatten()
    img_array = img_array.astype(np.float32)
    img_array = (img_array-128)/128
    img_array = np.array([img_array])
    pred = model.predict(img_array)[0]

    # Display neural result
    img_array2 = pred.reshape(128,128,3)
    img_array2 = (img_array2*128)+128
    img_array2 = img_array2.astype(np.uint8)
    img2 = Image.fromarray(img_array2, 'RGB')

    plt.imshow(img2)
    plt.show()
    
    score1 = np.sqrt(metrics.mean_squared_error(pred,img_array[0]))
    print(f"Out of Sample Score (RMSE): {score1}")   
    tt.append(score1)
    tt2.append(score1)
    print('=====================')
    print('=====================')
    i += 1
data = {'col_1': tt2}
df = pd.DataFrame.from_dict(data)
print(df.describe())

fig1, ax1 = plt.subplots()
ax1.set_title('box plot')
ax1.boxplot(tt2)
plt.scatter(sorted(tt2),range(0,len(tt2)), label='skitscat', color='k', s=25, marker="o")
plt.show()
tt2 = [0.042558003, 0.030652788, 0.042754516, 0.038435463, 0.038629595, 0.041084316, 0.035615653, 0.04272078, 0.033939373, 0.04175356, 0.041320685, 0.0368206, 0.042760883, 0.039931614, 0.03664131, 0.031103585, 0.038762517, 0.038499016,0.058573503, 0.066934064, 0.05464456, 0.07709787, 0.08883689, 0.07328239, 0.09117831, 0.07257624, 0.077166654, 0.059338007, 0.10371624, 0.042389914, 0.05010311, 0.07240847, 0.067032576, 0.062947355, 0.079062, 0.087789275, 0.06871186, 0.0634957, 0.052625958, 0.035812557, 0.078799985, 0.06328735, 0.06950112, 0.064118214, 0.060417272, 0.06591671, 0.09110638, 0.08182621, 0.070093654, 0.052168075, 0.05857365, 0.064667016, 0.075556725, 0.05015098, 0.061487824, 0.071132205, 0.07968406, 0.062037375, 0.068701394, 0.052525327, 0.07065824, 0.083880365, 0.09539992, 0.07136654, 0.08247659, 0.06425655, 0.06758966, 0.07027674, 0.05371886, 0.06454329, 0.038022883, 0.07525901, 0.05537062, 0.07471208, 0.05937753, 0.06530632, 0.09115084, 0.047818627, 0.06841827, 0.064744055, 0.09184977, 0.09172562, 0.07337593, 0.08758027, 0.06695, 0.060223594]

data = {'col_1': tt2}
df = pd.DataFrame.from_dict(data)
print(df.describe())

fig1, ax1 = plt.subplots()
ax1.set_title('box plot')
ax1.boxplot(tt2)

pictures = []
real = ['test_real_6.jpg', 'test_real_17.jpg', 'test_real_0.jpg', 'test_real_11.jpg', 'test_real_20.jpg', 'test_real_4.jpg', 'test_real_8.jpg', 'test_real_3.jpg', 'test_real_16.jpg', 'test_real_21.jpg', 'test_real_13.jpg', 'test_real_10.jpg', 'test_real_15.jpg', 'test_real_18.jpg', 'test_real_12.jpg', 'test_real_19.jpg', 'test_real_14.jpg', 'test_real_5.jpg']
label = []
for pic in real:
    pictures.append(pic)
    label.append(0)
    
fake = ['test_fake_23.jpg', 'test_fake_49.jpg', 'test_fake_71.jpg', 'test_fake_41.jpg', 'test_fake_50.jpg', 'test_fake_11.jpg', 'test_fake_27.jpg', 'test_fake_44.jpg', 'test_fake_16.jpg', 'test_fake_30.jpg', 'test_fake_68.jpg', 'test_fake_59.jpg', 'test_fake_70.jpg', 'test_fake_67.jpg', 'test_fake_38.jpg', 'test_fake_55.jpg', 'test_fake_43.jpg', 'test_fake_1.jpg', 'test_fake_22.jpg', 'test_fake_32.jpg', 'test_fake_75.jpg', 'test_fake_63.jpg', 'test_fake_12.jpg', 'test_fake_29.jpg', 'test_fake_8.jpg', 'test_fake_7.jpg', 'test_fake_18.jpg', 'test_fake_0.jpg', 'test_fake_62.jpg', 'test_fake_46.jpg', 'test_fake_51.jpg', 'test_fake_73.jpg', 'test_fake_20.jpg', 'test_fake_48.jpg', 'test_fake_4.jpg', 'test_fake_34.jpg', 'test_fake_58.jpg', 'test_fake_24.jpg', 'test_fake_57.jpg', 'test_fake_5.jpg', 'test_fake_74.jpg', 'test_fake_61.jpg', 'test_fake_15.jpg', 'test_fake_25.jpg', 'test_fake_13.jpg', 'test_fake_33.jpg', 'test_fake_66.jpg', 'test_fake_37.jpg', 'test_fake_65.jpg', 'test_fake_40.jpg', 'test_fake_28.jpg', 'test_fake_64.jpg', 'test_fake_39.jpg', 'test_fake_10.jpg', 'test_fake_35.jpg', 'test_fake_14.jpg', 'test_fake_69.jpg', 'test_fake_26.jpg', 'test_fake_42.jpg', 'test_fake_72.jpg', 'test_fake_47.jpg', 'test_fake_56.jpg', 'test_fake_19.jpg', 'test_fake_36.jpg', 'test_fake_6.jpg', 'test_fake_60.jpg', 'test_fake_54.jpg', 'test_fake_31.jpg']
for pic in fake:
    pictures.append(pic)
    label.append(1)
    
data = data = {'name': pictures, 'label':label}
df = pd.DataFrame.from_dict(data)
print(df.head())
print(df.tail())
y_fake = []

    
for pic in df.name.values.tolist():
    img = cv2.imread(pic)
    img = cv2.resize(img, dsize, Image.ANTIALIAS)
    y_fake.append(img)

print(len(y_fake))

i = 0
predictions = []
name = []
score = []
threshold_value = []
prediction_list = df.name.values.tolist()
label = df.label.values.tolist()
expected = []
output = []

print(prediction_list)
while i <len(prediction_list):
    img = y_fake[i]
    print(prediction_list[i])
    img = cv2.imread(prediction_list[i])
    img = cv2.resize(img, dsize)
    plt.imshow(img)
    img_array = np.asarray(img)
    print(img.size)

    #Display noisy image
    img2 = img_array.astype(np.uint8)
    img2 = Image.fromarray(img2, 'RGB')
    plt.imshow(img2)
    plt.show()
    # Present noisy image to auto encoder
    img_array = img_array.flatten()
    img_array = img_array.astype(np.float32)
    img_array = (img_array-128)/128
    img_array = np.array([img_array])
    pred = model.predict(img_array)[0]

    # Display neural result
    img_array2 = pred.reshape(128,128,3)
    img_array2 = (img_array2*128)+128
    img_array2 = img_array2.astype(np.uint8)
    img2 = Image.fromarray(img_array2, 'RGB')

    plt.imshow(img2)
    plt.show()
    
    score1 = np.sqrt(metrics.mean_squared_error(pred,img_array[0]))
    print(f"Out of Sample Score (RMSE): {score1}")   
    if score1 > threshold:
        value = 1
    else:
        value = 0
    if value == 1 and label[i] == 1:
        output.append(0)
    elif value == 0 and label[i] == 0:
        output.append(0)
    else:
        output.append(1)
    predictions.append(value)
    name.append(prediction_list[i])
    score.append(score1)
    threshold_value.append(threshold)
    expected.append(label[i])
    print('=====================')
    print('=====================')
    i += 1
print(predictions)
print(df.label.values.tolist())
data = data = {'name': pictures,'threshold':threshold,'score':score,'prediction':predictions,'label':expected, 'output':output}


df = pd.DataFrame.from_dict(data)
print(df)

error_df = pd.DataFrame({'reconstruction_error': score,
                        'true_class': expected})
error_df.describe()
fig = plt.figure()
ax = fig.add_subplot(111)
normal_error_df = error_df[(error_df['true_class']== 0)]
_ = ax.hist(normal_error_df.reconstruction_error.values, bins=10)
fig = plt.figure()
ax = fig.add_subplot(111)
fraud_error_df = error_df[error_df['true_class'] == 1]
_ = ax.hist(fraud_error_df.reconstruction_error.values, bins=10)
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)

fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();
precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
plt.plot(recall, precision, 'b', label='Precision-Recall curve')
plt.title('Recall vs Precision')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
plt.title('Precision for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.show()
plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve')
plt.title('Recall for different threshold values')
plt.xlabel('Reconstruction error')
plt.ylabel('Recall')
plt.show()
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

y_pred = df.prediction.values.tolist()
y_true = df.label.values.tolist()
y = np.array(y_true)
pred = np.array(y_pred)


print(accuracy_score(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
tn, fp, fn, tp = confusion_matrix(y_pred, y_true).ravel()
print((tn/len(y_pred), fp/len(y_pred), fn/len(y_pred), tp/len(y_pred)))
print(f1_score(y_true, y_pred, average='macro'))
print(mean_absolute_error(y_true, y_pred))
print(mean_squared_error(y_true, y_pred))




df = df.loc[df['output'] == 1]
print(df)
print(df.describe())

fig1, ax1 = plt.subplots()
ax1.set_title('box plot')
ax1.boxplot(df.score.values.tolist())
import seaborn as sns

LABELS = ["Normal", "FAKE"]
conf_matrix = confusion_matrix(error_df.true_class, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()





print(tt2)
import math
i = 0
print(min(tt))
mean_value = sum(tt)/len(tt)
print(mean_value)
while i < len(tt):
    tt[i] = (tt[i]-mean_value)**2
    i += 1
value_std = math.sqrt(sum(tt)/len(tt))
print(value_std)

#mean: 0.070140356581439
#std: 0.015058747266534056
t2 = []
t3 = []

i = 0
while i < len(tt):
    t2.append(t1[i]-2*value_std)
    t3.append(t1[i]+2*value_std)
    i += 1
print(t2)
print(t3)
print(min(t2))
TEST_PATH = '../input/deepfake-detection-challenge/test_videos/'
print(os.listdir(TEST_PATH))
print(path)
from PIL import Image, ImageEnhance 

def data_preparation(path,filename, meta, train = True):
    '''
    Helper function to visualize the 1st frame of the video by filename and metadata
    INPUT:
        filename - video filename
        meta - dataframe containing metadata.json
        train - indicates that the video is among train samples and the label can be retrived from metadata
    '''
    # get the 1st frame of the video
    image = get_frame(path+filename)
    image = white_balance(image)
    # Display the 1st frame of the video    
    # Extract the face with haar cascades
#    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dets = detector(image)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(image, 1)
    for det in dets:
        cv2.rectangle(image,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
        try:
            if len(cv2.rectangle(image,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)) == 1080:
                roi = image[det.top():det.bottom(),det.left():det.right()]    
            #image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                new_image = cv2.Laplacian(roi,cv2.CV_64F)
                f = np.fft.fft2(new_image)
                f_shift = np.fft.fftshift(f)
                f_complex = f_shift
                f_abs = np.abs(f_complex) + 1 # lie between 1 and 1e6
                f_bounded = 20 * np.log(f_abs)
                f_img = 255 * f_bounded / np.max(f_bounded)
                f_img = f_img.astype(np.uint8)
                #prediction_list.append(str(filename))
                return(f_img)
        except:
            pass


i = 0
path = '../input/deepfake-detection-challenge/test_videos/'
submission_path = os.listdir(path)
print(submission_path)
prediction_list = []

while i < len(submission_path):
    print(submission_path[i])
    laplacian_answer = data_preparation(path,submission_path[i], meta)
    try:
        cv2.imwrite('submission_'+submission_path[i]+'.jpg', laplacian_answer) 
        prediction_list.append('submission_'+submission_path[i]+'.jpg')
        plt.imshow(laplacian_answer)
        plt.show()
    except:
        pass
    i += 1

print(prediction_list)
print(prediction_list)
y_fake = []

    
for pic in prediction_list:
    print(pic)
    img = cv2.imread(pic)
    print(img)
    img = cv2.resize(img, dsize, Image.ANTIALIAS)
    y_fake.append(img)

print(len(y_fake))
print(prediction_list)

i = 0

predictions = []
while i <len(prediction_list):
    img = y_fake[i]
    print(prediction_list[i])
    img = cv2.imread(prediction_list[i])
    img = cv2.resize(img, dsize)
    plt.imshow(img)
    img_array = np.asarray(img)
    print(img.size)

    #Display noisy image
    img2 = img_array.astype(np.uint8)
    img2 = Image.fromarray(img2, 'RGB')
    plt.imshow(img2)
    plt.show()
    # Present noisy image to auto encoder
    img_array = img_array.flatten()
    img_array = img_array.astype(np.float32)
    img_array = (img_array-128)/128
    img_array = np.array([img_array])
    pred = model.predict(img_array)[0]

    # Display neural result
    img_array2 = pred.reshape(128,128,3)
    img_array2 = (img_array2*128)+128
    img_array2 = img_array2.astype(np.uint8)
    img2 = Image.fromarray(img_array2, 'RGB')

    plt.imshow(img2)
    plt.show()
    
    score1 = np.sqrt(metrics.mean_squared_error(pred,img_array[0]))
    print(f"Out of Sample Score (RMSE): {score1}")   
    if score1 > threshold:
        predictions.append(1)
    else:
        predictions.append(0)
    print('=====================')
    print('=====================')
    i += 1


submission_df = pd.DataFrame({"filename": prediction_list, "label": predictions})
submission_df.to_csv("submission.csv", index=False)


print(predictions)
print(submission_df.tail(20))







t1 = [0.041331466, 0.024498282, 0.04118593, 0.038716756, 0.037078038, 0.03631261, 0.03412535, 0.04222116, 0.039362844, 0.04294556, 0.0386346, 0.03914516, 0.041025102, 0.036734298, 0.025888393, 0.028660871, 0.041659564, 0.039061926]
t2 = []
t3 = []

i = 0
while i < len(t1):
    t2.append(t1[i]-2*0.005346882783381586)
    t3.append(t1[i]+2*0.005346882783381586)
    i += 1
print(t2)
print(t3)
print(max(t3))
t1 = [0.058573503, 0.066934064, 0.05464456, 0.07709787, 0.08883689, 0.07328239, 0.09117831, 0.07257624, 0.077166654, 0.059338007, 0.10371624, 0.042389914, 0.05010311, 0.07240847, 0.067032576, 0.062947355, 0.079062, 0.087789275, 0.06871186, 0.0634957, 0.052625958, 0.035812557, 0.078799985, 0.06328735, 0.06950112, 0.064118214, 0.060417272, 0.06591671, 0.09110638, 0.08182621, 0.070093654, 0.052168075, 0.05857365, 0.064667016, 0.075556725, 0.05015098, 0.061487824, 0.071132205, 0.07968406, 0.062037375, 0.068701394, 0.052525327, 0.07065824, 0.083880365, 0.09539992, 0.07136654, 0.08247659, 0.06425655, 0.06758966, 0.07027674, 0.05371886, 0.06454329, 0.038022883, 0.07525901, 0.05537062, 0.07471208, 0.05937753, 0.06530632, 0.09115084, 0.047818627, 0.06841827, 0.064744055, 0.09184977, 0.09172562, 0.07337593, 0.08758027, 0.06695, 0.060223594]
t2 = []
t3 = []

i = 0
while i < len(t1):
    t2.append(t1[i]-2*0.013818416343404723)
    t3.append(t1[i]+2*0.013818416343404723)
    i += 1
print(t2)
print(t3)
print(min(t2))
print(min(t1))







img = y_fake[0]
print(test_pictures[0])
img = cv2.imread(pic)
img = cv2.resize(img, dsize)
plt.imshow(img)
img_array = np.asarray(img)
print(img.size)

#Display noisy image
img2 = img_array.astype(np.uint8)
img2 = Image.fromarray(img2, 'RGB')
plt.imshow(img2)
plt.show()
# Present noisy image to auto encoder
img_array = img_array.flatten()
img_array = img_array.astype(np.float32)
img_array = (img_array-128)/128
img_array = np.array([img_array])
pred = model.predict(img_array)[0]

# Display neural result
img_array2 = pred.reshape(128,128,3)
img_array2 = (img_array2*128)+128
img_array2 = img_array2.astype(np.uint8)
img2 = Image.fromarray(img_array2, 'RGB')

plt.imshow(img2)
plt.show()
img = y[1]
print(test_pictures[1])
img = cv2.imread(pic)
img = cv2.resize(img, dsize)
plt.imshow(img)
img_array = np.asarray(img)
print(img.size)

#Display noisy image
img2 = img_array.astype(np.uint8)
img2 = Image.fromarray(img2, 'RGB')
plt.imshow(img2)
plt.show()
# Present noisy image to auto encoder
img_array = img_array.flatten()
img_array = img_array.astype(np.float32)
img_array = (img_array-128)/128
img_array = np.array([img_array])
pred = model.predict(img_array)[0]

# Display neural result
img_array2 = pred.reshape(128,128,3)
img_array2 = (img_array2*128)+128
img_array2 = img_array2.astype(np.uint8)
img2 = Image.fromarray(img_array2, 'RGB')

plt.imshow(img2)
plt.show()


print(x.shape)
pred = model.predict(x[0])

pred = pred.reshape(128,128,3)
pred = (pred*128)+128
pred = pred.astype(np.uint8)
pred = Image.fromarray(pred, 'RGB')


plt.imshow(pred)
plt.show()
from PIL import Image

def data_preparation(i,title,filename, meta, train = True):
    '''
    Helper function to visualize the 1st frame of the video by filename and metadata
    INPUT:
        filename - video filename
        meta - dataframe containing metadata.json
        train - indicates that the video is among train samples and the label can be retrived from metadata
    '''
    # get the 1st frame of the video

    image = get_frame(filename)
    image = white_balance(image)

    # Display the 1st frame of the video
    
    # Extract the face with haar cascades
#    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dets = detector(image)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(image, 1)
    for det in dets:
        cv2.rectangle(image,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
        if len(cv2.rectangle(image,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)) == 1080:
            roi = image[det.top():det.bottom(),det.left():det.right()]            
            #image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            new_image = cv2.Laplacian(roi,cv2.CV_64F)
            f = np.fft.fft2(new_image)
            f_shift = np.fft.fftshift(f)
            f_complex = f_shift
            f_abs = np.abs(f_complex) + 1 # lie between 1 and 1e6
            f_bounded = 20 * np.log(f_abs)
            f_img = 255 * f_bounded / np.max(f_bounded)
            f_img = f_img.astype(np.uint8)
            w, h = 185, 186
            img = Image.fromarray(f_img, 'RGB')
            img.save(title+str(i)+'.png')
            img.show()
        else:
            pass
from PIL import Image

def training_data_preparation(filename):
    '''
    Helper function to visualize the 1st frame of the video by filename and metadata
    INPUT:
        filename - video filename
        meta - dataframe containing metadata.json
        train - indicates that the video is among train samples and the label can be retrived from metadata
    '''
    # get the 1st frame of the video

    image = cv2.imread(filename,1)
    image = white_balance(image)

    # Display the 1st frame of the video
    
    # Extract the face with haar cascades
#    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dets = detector(image)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(image, 1)
    for det in dets:
        cv2.rectangle(image,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
        if len(cv2.rectangle(image,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)) == 1080:
            roi = image[det.top():det.bottom(),det.left():det.right()]            
            #image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            new_image = cv2.Laplacian(roi,cv2.CV_64F)
            f = np.fft.fft2(new_image)
            f_shift = np.fft.fftshift(f)
            f_complex = f_shift
            f_abs = np.abs(f_complex) + 1 # lie between 1 and 1e6
            f_bounded = 20 * np.log(f_abs)
            f_img = 255 * f_bounded / np.max(f_bounded)
            f_img = f_img.astype(np.uint8)
            w, h = 185, 186
            img = Image.fromarray(f_img, 'RGB')
            img.save(filename)
            img.show()
        else:
            pass
#preparing the training and testing data

real_train_sample_video = list(meta.loc[meta.label=='REAL'].index)
print(len(real_train_sample_video))
fake_train_sample_video = list(meta.loc[meta.label=='FAKE'].index)
print(len(fake_train_sample_video))
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
import numpy as np
import scipy.misc
import numpy.random as rng
import math

import cv2
import PIL
import os
from pathlib import Path
import glob
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave
from keras.preprocessing.image import load_img, array_to_img, img_to_array
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Input
from keras.optimizers import SGD, Adam, Adadelta, Adagrad
from keras import backend as K
from sklearn.model_selection import train_test_split
np.random.seed(111)
from subprocess import check_output

import cv2
import keras
import PIL
import os
from pathlib import Path
import glob
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave
from keras.preprocessing.image import load_img, array_to_img, img_to_array
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Input
from keras.optimizers import SGD, Adam, Adadelta, Adagrad
from keras import backend as K
from sklearn.model_selection import train_test_split
np.random.seed(111)
from subprocess import check_output
import numpy as np
import math
import tensorflow as tf
import nibabel as nib
import numpy as np
from keras.layers import Input,Dense,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras import backend as K
import scipy.misc
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras.models import model_from_json

import numpy as np
import math
import tensorflow as tf
import nibabel as nib
import numpy as np
from keras.layers import Input,Dense,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras import backend as K
import scipy.misc
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras.models import model_from_json
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


#preparing the training and testing data


print(meta.loc[meta.label=='REAL'].describe)
i = 0
while i < len(real_train_sample_video):
    vidcap = cv2.VideoCapture('../input/deepfake-detection-challenge/train_sample_videos/'+str(real_train_sample_video[i]))
    success,image = vidcap.read()
    count = 0 
    while success:
        cv2.imwrite('train_real_'+str(real_train_sample_video[i])+"_%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1
    i += 1

print(os.listdir('.'))
filename = os.listdir()
print(len(filename))
i = 0
default = []
while i < len(train_sample_video):
    if filename[i][0:5] == 'train':
        try:
            print(filename[i])
            training_data_preparation(filename[i])
        except:
            print(filename[i])
            default.append(filename[i])
            pass
    i += 1
print(len(default))
print('train real')
img = plt.imread('train_real_cppdvdejkc.mp4_136.jpg')
imgplot = plt.imshow(img)
plt.show()

#preparing the training and testing data


print(meta.loc[meta.label=='FAKE'].describe)
train_sample_video = list(meta.loc[meta.label=='FAKE'].index)

i = 0
while i < len(train_sample_video[0:5]):
    vidcap = cv2.VideoCapture('../input/deepfake-detection-challenge/train_sample_videos/'+str(train_sample_video[i]))
    success,image = vidcap.read()
    count = 0 
    while success:
        cv2.imwrite('test_fake_'+str(train_sample_video[i])+"_%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1
    i += 1
filename = os.listdir()
print(len(filename))
i = 0
default = []
while i < len(train_sample_video):
    if filename[i][0:4] == 'test':
        try:
            print(filename[i])
            training_data_preparation(filename[i])
        except:
            print(filename[i])
            default.append(filename[i])
            pass
    i += 1
print(len(default))
print('test fake')
img = plt.imread('test_fake_agrmhtjdlk.mp4_213.jpg')
imgplot = plt.imshow(img)
plt.show()





#preparing the training dataset
filename = os.listdir()
print(len(filename))
i = 0
while i < len(train_sample_video):
    print(filename[i])
    if filename[i][0:5] == 'train':
        try:
            print('')
            #training_data_preparation(filename[i])
        except:
            print(filename[i])
            pass
    i += 1
#preparing the testing dataset
i = 0
while i <  len(fake_test_sample_video):
    data_preparation(i,'test_real_','../input/deepfake-detection-challenge/train_sample_videos/'+str(fake_test_sample_video[i]), meta)    
    i += 1

fake_train_sample_video = list(meta.loc[meta.label=='FAKE'].index)[0:20]
i = 0
#preparing the training dataset
while i < 20:#len(fake_train_sample_video):
    data_preparation(i,'test_fake_','../input/deepfake-detection-challenge/train_sample_videos/'+str(fake_train_sample_video[i]), meta)
    i += 1

print(os.listdir())
import os
data = os.listdir()
train_pictures = []
test_pictures = []
labels = []
data_type = []

i = 0
while i < len(data):
    if data[i][0:5] == 'train':
        train_pictures.append(data[i])
        labels.append(0)
        data_type.append('TRAIN')
    elif data[i][0:4] == 'test':
        test_pictures.append(data[i])
        if data[i][5:9] == 'fake':
            labels.append(0)
            data_type.append('TEST')
        elif data[i][5:9] == 'real':
            labels.append(1)
            data_type.append('TEST')
    else:
        pass
    i += 1

pictures = train_pictures + test_pictures

input_img = Input(shape=(192,192,3))

from pathlib import Path
train_images = train_pictures

X = []
Y = []

i = 0
for img in train_images:
    try:
        img = load_img(img, grayscale=False,target_size=(192, 192, 3))
        img = img_to_array(img).astype('float32')/255.
        X.append(img)
    except:
        i += 1
        pass
print(i)

for img in train_images:
    try:
        img = cv2.imread(img,1)
        img = cv2.resize(img,((192, 192)))
        img = img_to_array(img).astype('float32')/255.
        Y.append(img)
    except:
        pass
X = np.array(X[0:1000])
Y = np.array(Y[0:1000])

print("Size of X : ", X.shape)
#print("Size of Y : ", Y.shape)  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=111)
x_train = X_train
x_test = X_test
# import libraries
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import matplotlib.pyplot as plt
from keras.models import load_model

# define input shape
input_img = Input(shape=(192, 192, 3))

# encoding dimension
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# decoding dimension
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((4, 4))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
# build model
autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')



#autoencoder = Model(input_img, decoded)
#utoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
decoded_imgs = autoencoder.predict(x_test)
n = 4
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i + 1].reshape(192, 192, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i + 1].reshape(192, 192,3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from pylab import rcParams

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
nb_epoch = 10
batch_size = 320

#autoencoder = build_autoencoder()


autoencoder.compile(optimizer='sgd', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])



history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1).history
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
input_img = Input(shape=(784,))
input_img2 = Input(shape=(32,))
X_train = X_train.reshape(len(X_train), np.prod(X_train.shape[1:]))
X_test = X_test.reshape(len(X_test), np.prod(X_test.shape[1:]))
encoded = Dense(units=128, activation='relu')(input_img)
encoded = Dense(units=64, activation='relu')(encoded)
encoded = Dense(units=32, activation='relu')(encoded)

decoded = Dense(units=64, activation='relu')(encoded)
decoded = Dense(units=128, activation='relu')(decoded)
decoded = Dense(units=784, activation='sigmoid')(decoded)
autoencoder = Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=512,
                shuffle=True,
                validation_data=(X_test, X_test))
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');
plt.show()
predictions = autoencoder.predict(x_test)
print(predictions)
print(predictions[0])
%matplotlib inline 
from matplotlib import pyplot as plt



a = np.expand_dims(predictions[0], axis=0)  # or axis=1
plt.imshow(a)
plt.show()
plt.imshow(predictions, interpolation='nearest')
plt.show()
Y_test = df.loc[df['target_class'] == 0]
y_test = Y_test['target_class']

predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
print(mse[25])

error_df = pd.DataFrame({'reconstruction_error': mse, 'true_class': y_test})

print(error_df.describe())
Y_test = df.loc[df['target_class'] == 1]
y_test = Y_test['target_class']

predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
print(mse[0])

nb_epoch = 1000
batch_size = 320

autoencoder.compile(optimizer='sgd', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

history = autoencoder.fit(x_train, x_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history
autoencoder = build_autoencoder()

autoencoder.compile(metrics=['accuracy'],
                    loss='mean_squared_error',
                    optimizer='adam')


tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

autoencoder.summary()


autoencoder = autoencoder.fit(x_train, x_train,
                    epochs=10,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    verbose=1)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');
plt.show()
predictions = autoencoder.predict(X_test)
print(type(predictions))


def visualize(img,encoder,decoder):
    """Draws original, encoded and decoded images"""
    # img[None] will have shape of (1, 32, 32, 3) which is the same as the model input
    code = encoder.predict(img[None])[0]
    reco = decoder.predict(code[None])[0]

    plt.subplot(1,3,1)
    plt.title("Original")
    show_image(img)

    plt.subplot(1,3,2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1]//2,-1]))

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()

for i in range(5):
    img = X_test[i]
    visualize(img,encoder,decoder)



