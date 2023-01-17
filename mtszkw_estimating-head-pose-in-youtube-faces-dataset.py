import os

import cv2

import glob

import imageio

import numpy as np

import pandas as pd



import matplotlib

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from matplotlib.patches import Rectangle

from matplotlib.animation import FuncAnimation



from IPython.display import Image, display



%matplotlib inline
class InputVideo():

    def __init__(self, video_file, video_name):

        self.name_ = video_name

        self.color_images_ = video_file['colorImages']

        self.bounding_box_ = video_file['boundingBox']

        self.landmarks_2D_ = video_file['landmarks2D']

        self.landmarks_3D_ = video_file['landmarks3D']

        

    def name(self):

        return self.name_

    

    def frames(self):

        return self.color_images_

    

    def length(self):

        return self.color_images_.shape[3]

    

    def frame(self, i):

        return self.color_images_[:, :, :, i]

    

    def landmarks_2D(self):

        return self.landmarks_2D_

    

    def landmarks_3D(self):

        return self.landmarks_3D_

    

    def bounding_box(self):

        return self.bounding_box_
class HeadPoseDetector():

    def detect(self, im, landmarks_2d, landmarks_3d):

        h, w, c = im.shape

        K = np.array([[w, 0, w/2],

                      [0, w, h/2],

                      [0, 0, 1]], dtype=np.double)

        dist_coeffs = np.zeros((4,1)) 



        (_, R, t) = cv2.solvePnP(landmarks_3d, landmarks_2d, K, dist_coeffs)

        return R, t, K, dist_coeffs
base_path = '/kaggle/input/youtube-faces-with-facial-keypoints/'

df_videos = pd.read_csv(base_path + 'youtube_faces_with_keypoints_large.csv')



# https://www.kaggle.com/selfishgene/exploring-youtube-faces-with-keypoints-dataset

# Create a dictionary that maps videoIDs to full file paths

npz_file_paths = glob.glob(base_path + 'youtube_faces_*/*.npz')

video_ids = [x.split('/')[-1].split('.')[0] for x in npz_file_paths]



full_video_paths = {}

for video_id, full_path in zip(video_ids, npz_file_paths):

    full_video_paths[video_id] = full_path



# Remove from the large csv file all videos that weren't uploaded yet

df_videos = df_videos.loc[df_videos.loc[:,'videoID'].isin(full_video_paths.keys()), :].reset_index(drop=True)
num_samples = 3

np.random.seed(11)

sample_indices = np.random.choice(df_videos.index, size=num_samples, replace=False)

sample_video_ids = df_videos.loc[sample_indices, 'videoID']



sample_videos = []

for i, video_id in enumerate(sample_video_ids):

    sv = InputVideo(np.load(full_video_paths[video_id]), video_id)

    sample_videos.append(sv)
for i, video in enumerate(sample_videos):

    title = video.name()

    frames = [video.frame(f) for f in range(video.length())]

    

    kwargs_write = {'fps':1.0, 'quantizer':'nq'}

    imageio.mimsave(f'./{title}_orig.gif', frames, fps=24)

    display(Image(url=f'./{title}_orig.gif'))
face_landmark_lines = [

    [1, 5], [5, 7], [7, 9], # right-side jaw

    [9, 11], [11, 13], [13, 17], # left-side jaw

    [17, 25], [25, 20], [20, 1], # upper head

    [28, 31], [28, 25], [28, 20], [31, 32], [31, 36], # nose

    [32, 52], [36, 52], # nose-mouth

    [52, 49], [49, 58], [58, 55], [55, 52], # mouth

    [6, 49], [55 ,12], [58, 9], # mouth-jaw

    [46, 17], # left eye - jaw

    [1, 37], # right eye - jaw 

    [37, 38], [38, 39], [39, 40], [40, 41], [41, 42], [42, 37], # right eye

    [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 43], # left eye

    [28, 40], [28, 43]

]
def find_bounding_box(landmarks_x, landmarks_y):

    return min(landmarks_x), min(landmarks_y), max(landmarks_x), max(landmarks_y)
def process_frame(frame, landmarks_2D, landmarks_3D):

    processed_img = np.array(frame)

    overlay = processed_img.copy()

    

    # Face grid

    for mark_id1, mark_id2 in face_landmark_lines:

        pt1 = ( int(landmarks_2D[mark_id1-1,0]), int(landmarks_2D[mark_id1-1,1]) )

        pt2 = ( int(landmarks_2D[mark_id2-1,0]), int(landmarks_2D[mark_id2-1,1]) )

        cv2.line(overlay, pt1, pt2, color=(255,255,255))

        

    # Bounding box

    x0, y0, x1, y1 = map(int, find_bounding_box(landmarks_2D[:, 0], landmarks_2D[:, 1]))

    cv2.rectangle(overlay, (x0, y0), (x1, y1), color=(255, 255, 255))



    # Transparency

    alpha = 0.3

    processed_img = cv2.addWeighted(overlay, alpha, processed_img, 1 - alpha, 0)

    

    # Draw pose change

    image_center = (int(frame.shape[1]/2), int(frame.shape[0]/2))

    nose_tip = landmarks_2D[33]

#     (R, t, K, dist_coeffs) = HeadPoseDetector().detect(frame, landmarks_2D, landmarks_3D)

    cv2.line(processed_img, image_center, (int(nose_tip[0]), int(nose_tip[1])), color=(255,0,0))

    

    return processed_img
for i, video in enumerate(sample_videos):

    title = video.name()

    

    processed_imgs = [process_frame(video.frame(f),

                                    video.landmarks_2D()[:, :, f],

                                    video.landmarks_3D()[:, :, f]) for f in range(video.length())]



    imageio.mimsave(f'./{title}_2d.gif', processed_imgs, fps=24)

    display(Image(url=f'./{title}_2d.gif'))