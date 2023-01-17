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

df_videos = df_videos.loc[df_videos.loc[:,'videoID']

                          .isin(full_video_paths.keys()), :].reset_index(drop=True)
num_samples = 3

np.random.seed(25)



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
def add_mask_overlay(overlay, landmarks_2D):

    c = (10, 10, 10) # color

    landmark_vertex_ids = [

        3, 30, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4

    ]

    

    vertices = np.array([[int(landmarks_2D[i-1,0]),

                          int(landmarks_2D[i-1,1])] for i in landmark_vertex_ids])

    

    cv2.fillConvexPoly(overlay, np.int32([vertices]), color=c, lineType=cv2.LINE_AA)

    

    mask_link_left_1 = 15

    mask_link_left_2 = 17

    cv2.line(overlay,

             (int(landmarks_2D[mask_link_left_1-1,0]),

              int(landmarks_2D[mask_link_left_1-1,1])),

             (int(landmarks_2D[mask_link_left_2-1,0]),

              int(landmarks_2D[mask_link_left_2-1,1])),

             color=c,

             thickness=1)

    

    mask_link_right_1 = 1

    mask_link_right_2 = 3

    cv2.line(overlay,

             (int(landmarks_2D[mask_link_right_1-1,0]),

              int(landmarks_2D[mask_link_right_1-1,1])),

             (int(landmarks_2D[mask_link_right_2-1,0]),

              int(landmarks_2D[mask_link_right_2-1,1])),

             color=c,

             thickness=1)
def process_frame(frame, landmarks_2D):

    processed_img = np.array(frame)



    mask_overlay = np.zeros(processed_img.shape,dtype=np.uint8)

    mask_overlay.fill(255)

    add_mask_overlay(mask_overlay, landmarks_2D)

    

    # Smooth mask

    blurred_img = cv2.GaussianBlur(mask_overlay, (21, 21), 0)

    mask_mask = np.zeros(mask_overlay.shape, np.uint8)

    gray_mask = cv2.cvtColor(mask_overlay, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray_mask, 60, 255, cv2.THRESH_BINARY)[1]

    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(mask_mask, contours, -1, (255,255,255),5)

    mask_overlay = np.where(mask_mask==np.array([255, 255, 255]), blurred_img, mask_overlay)

    

    # Apply overlay

    processed_img = cv2.bitwise_and(processed_img, mask_overlay)

        

    return processed_img
for i, video in enumerate(sample_videos):

    title = video.name()

    

    processed_imgs = [

        process_frame(video.frame(f),

                      video.landmarks_2D()[:, :, f])

                      for f in range(video.length())

    ]



    imageio.mimsave(f'./{title}_mask.gif', processed_imgs, fps=24)

    display(Image(url=f'./{title}_mask.gif'))