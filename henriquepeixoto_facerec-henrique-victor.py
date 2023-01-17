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





import matplotlib.pylab as plt

import cv2

plt.style.use('ggplot')

from IPython.display import Video

from IPython.display import HTML

from tqdm import tqdm



from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing import MinMaxScaler, Imputer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Imputer

train_dir = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'

train_video_files = [train_dir + x for x in os.listdir(train_dir)]

test_dir = '/kaggle/input/deepfake-detection-challenge/test_videos/' 

test_video_files = [test_dir + x for x in os.listdir(test_dir)]
df_train = pd.read_json('/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json').transpose()

df_train.head()
df_train.groupby('label')['label'].count().plot(figsize=(15, 5), kind='bar', title='Distribution of Labels in the Training Set')

plt.show()
df_train['label'].value_counts(normalize=True)
target = []



for i in df_train['label']:

    if i == 'REAL':

        target.append(1)

    else:

        target.append(0)

        

df_train['Target'] = target
#Real == 1 Fake == 0

df_train.drop(['label', 'split'], axis=1, inplace=True)

df_train.head()
import cv2 as cv



def display_image_from_video(video_path):

    '''

    input: video_path - path for video

    process:

    1. perform a video capture from the video

    2. read the image

    3. display the image

    '''

    capture_image = cv.VideoCapture(video_path) 

    ret, frame = capture_image.read()

    fig = plt.figure(figsize=(10,10))

    ax = fig.add_subplot(111)

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    return    ax.imshow(frame)    
display_image_from_video('/kaggle/input/deepfake-detection-challenge/train_sample_videos/ebebgmtlcu.mp4')
def framecap(path):    

    cam = cv2.VideoCapture(path) 



    try: 



        # creating a folder named data 

        if not os.path.exists('/kaggle/input/deepfake-detection-challenge/train_images'): 

            os.makedirs('/kaggle/input/deepfake-detection-challenge/train_images') 



    # if not created then raise error 

    except OSError: 

        print ('Error: Creating directory of data') 



    # frame 

    currentframe = 0



    while(True): 



        # reading from frame 

        ret,frame = cam.read() 



        if ret: 

            # if video is still left continue creating images 

            name = './kaggle/input/deepfake-detection-challenge/train_images' + str(currentframe) + '.jpg'

            print ('Creating...' + name) 



            # writing the extracted images 

            cv2.imwrite(name, frame) 



            # increasing counter so that it will 

            # show how many frames are created 

            currentframe += 1

        else: 

            break



    # Release all space and windows once done 

    cam.release() 

    cv2.destroyAllWindows() 
framecap('/kaggle/input/deepfake-detection-challenge/train_sample_videos/ebebgmtlcu.mp4')